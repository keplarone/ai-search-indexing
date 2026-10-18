"""Parser and lead-file tests for the agency miner. No browser is launched."""

from __future__ import annotations

import json

import lead_miner


def test_people_are_read_from_lines_and_json_ld() -> None:
    text = (
        "Jane Alvarez\n"
        "Managing Director\n"
        "John Smith - VP of SEO\n"
        "Chief Technology Officer\n"
        "Intern coordinator"
    )
    people = lead_miner.people_from_text(text)
    assert ("Jane Alvarez", "Managing Director") in people
    assert ("John Smith", "VP of SEO") in people
    assert ("", "CTO") in people
    assert all(title != "Intern coordinator" for _name, title in people)

    payload = {
        "@type": "Organization",
        "employee": [
            {"@type": "Person", "name": "Ada Lovelace", "jobTitle": "Chief Technology Officer"},
            {"@type": "Person", "name": "Grace Hopper", "jobTitle": "Copy lead"},
        ],
    }
    assert lead_miner.people_from_json_ld(payload) == [("Ada Lovelace", "CTO")]


def test_missing_title_and_bad_url_become_empty() -> None:
    assert lead_miner.canonical_title("office manager") == ""
    assert lead_miner.canonical_website("") == ""
    assert lead_miner.canonical_website("javascript:alert(1)") == ""
    assert lead_miner.canonical_website("https://www.Example.com/team") == "https://example.com"
    assert lead_miner._is_agency_candidate("https://www.linkedin.com/company/acme") is False
    assert lead_miner._is_agency_candidate("https://acme.example/about") is True


def test_field_classifier_ignores_unknown_controls() -> None:
    assert lead_miner.classify_field({"tag": "textarea", "type": "", "name": "comment"}) == "message"
    assert lead_miner.classify_field({"tag": "input", "type": "email", "name": "email"}) == "email"
    assert lead_miner.classify_field({"tag": "input", "type": "text", "name": "phone"}) == "phone"
    assert lead_miner.classify_field({"tag": "input", "type": "text", "name": "favorite_color"}) == ""


def test_citation_message_uses_the_captured_company() -> None:
    message = lead_miner.forensic_message(
        {
            "company_name": "Northwind",
            "website_url": "https://northwind.example",
            "location": "New York",
        }
    )
    assert "Northwind" in message
    assert "https://northwind.example" in message
    assert "New York" in message
    assert "citation leak" in message


def test_leads_are_appended_atomically_and_deduped(tmp_path) -> None:
    path = tmp_path / "leads.json"
    store = lead_miner.LeadStore(path)
    store.append(
        {
            "company_name": "Northwind",
            "website_url": "https://northwind.example",
            "professional_title": "",
            "contact_name": "",
        }
    )
    assert store.contains("https://www.northwind.example/contact") is True
    assert store.run_count == 1
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved[0]["professional_title"] == ""
    assert saved[0]["company_name"] == "Northwind"

    path.write_text("{", encoding="utf-8")
    assert store._read() == []
    assert path.with_suffix(".json.bak").exists()
