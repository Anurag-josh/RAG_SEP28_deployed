"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: Maintain the whitelist of trusted domains for educational and engineering sources.
Dependencies: None
"""

from typing import Set

# Whitelist of trusted educational, research, and technical documentation domains.
from typing import Set, Dict, Any, Optional

# Whitelist of trusted educational, research, and technical documentation domains.
TRUSTED_DOMAINS: Set[str] = {
    "geeksforgeeks.org",
    "nptel.ac.in",
    "ocw.mit.edu",
    "stanford.edu",
    "harvard.edu",
    "wikipedia.org",
    "learn.microsoft.com",
    "docs.oracle.com",
    "docs.python.org",
    "developer.mozilla.org",
    "github.com",
    "w3schools.com",
    "scipy.org",
    "numpy.org",
    "pandas.pydata.org",
    "tutorialspoint.com",
    "coursera.org",
    "edx.org",
    "khanacademy.org",
    "stackoverflow.com"
}

# Trust metadata hierarchy & categorization
# Tier 1: Official documentation / government / primary authoritative technical sources
# Tier 2: Academic institutions / established technical repositories
# Tier 3: Educational platforms & developer reference portals
TRUSTED_SOURCE_METADATA: Dict[str, Dict[str, Any]] = {
    "docs.python.org": {"tier": 1, "tier_name": "Tier 1 (Official Documentation)", "category": "Official Language Docs", "name": "Python Documentation"},
    "developer.mozilla.org": {"tier": 1, "tier_name": "Tier 1 (Official Documentation)", "category": "Official Web Docs", "name": "MDN Web Docs"},
    "learn.microsoft.com": {"tier": 1, "tier_name": "Tier 1 (Official Documentation)", "category": "Official Tech Docs", "name": "Microsoft Learn"},
    "docs.oracle.com": {"tier": 1, "tier_name": "Tier 1 (Official Documentation)", "category": "Official Enterprise Docs", "name": "Oracle Documentation"},
    "scipy.org": {"tier": 1, "tier_name": "Tier 1 (Official Documentation)", "category": "Official Scientific Computing", "name": "SciPy Documentation"},
    "numpy.org": {"tier": 1, "tier_name": "Tier 1 (Official Documentation)", "category": "Official Scientific Computing", "name": "NumPy Documentation"},
    "pandas.pydata.org": {"tier": 1, "tier_name": "Tier 1 (Official Documentation)", "category": "Official Data Science Docs", "name": "Pandas Documentation"},
    
    "stanford.edu": {"tier": 2, "tier_name": "Tier 2 (Academic & Research)", "category": "University Research", "name": "Stanford University"},
    "harvard.edu": {"tier": 2, "tier_name": "Tier 2 (Academic & Research)", "category": "University Research", "name": "Harvard University"},
    "ocw.mit.edu": {"tier": 2, "tier_name": "Tier 2 (Academic & Research)", "category": "Academic Courseware", "name": "MIT OpenCourseWare"},
    "nptel.ac.in": {"tier": 2, "tier_name": "Tier 2 (Academic & Research)", "category": "National Engineering Education", "name": "NPTEL India"},
    "github.com": {"tier": 2, "tier_name": "Tier 2 (Academic & Research)", "category": "Source Code & Spec Docs", "name": "GitHub"},

    "geeksforgeeks.org": {"tier": 3, "tier_name": "Tier 3 (Educational Reference)", "category": "Computer Science Reference", "name": "GeeksforGeeks"},
    "wikipedia.org": {"tier": 3, "tier_name": "Tier 3 (Educational Reference)", "category": "General Knowledge Encyclopedia", "name": "Wikipedia"},
    "w3schools.com": {"tier": 3, "tier_name": "Tier 3 (Educational Reference)", "category": "Web Technology Reference", "name": "W3Schools"},
    "tutorialspoint.com": {"tier": 3, "tier_name": "Tier 3 (Educational Reference)", "category": "Computer Science Tutorials", "name": "TutorialsPoint"},
    "coursera.org": {"tier": 3, "tier_name": "Tier 3 (Educational Reference)", "category": "Online Education", "name": "Coursera"},
    "edx.org": {"tier": 3, "tier_name": "Tier 3 (Educational Reference)", "category": "Online Education", "name": "edX"},
    "khanacademy.org": {"tier": 3, "tier_name": "Tier 3 (Educational Reference)", "category": "K-12 & STEM Learning", "name": "Khan Academy"},
    "stackoverflow.com": {"tier": 3, "tier_name": "Tier 3 (Educational Reference)", "category": "Developer Community Q&A", "name": "Stack Overflow"}
}

def clean_domain(domain: str) -> str:
    """
    Normalizes a domain string by stripping whitespace, trailing dots, ports, and 'www.' prefix.
    """
    if not domain:
        return ""
    d = domain.lower().strip().rstrip(".")
    # Remove port if present
    if ":" in d:
        d = d.split(":", 1)[0]
    # Remove leading www. for uniform domain matching
    if d.startswith("www."):
        d = d[4:]
    return d

def is_domain_trusted(domain: str) -> bool:
    """
    Validates that a domain string is valid and non-empty.
    Static domain whitelisting is disabled in favor of dynamic broad web search.
    """
    cleaned = clean_domain(domain)
    return bool(cleaned)

def get_source_metadata(domain: str) -> Dict[str, Any]:
    """
    Retrieves neutral source metadata for a given domain.
    """
    cleaned = clean_domain(domain)
    if not cleaned:
        return {"category": "Web Source", "name": "Web Source"}
    return {"category": "Web Source", "name": cleaned.capitalize()}

if __name__ == "__main__":
    print("Testing trusted_sources.py...")
    test_domains = [
        "stanford.edu",
        "wikipedia.org",
        "geeksforgeeks.org",
        "programiz.com",
        "example.com"
    ]
    for d in test_domains:
        valid = is_domain_trusted(d)
        meta = get_source_metadata(d)
        print(f"Domain: {d:<25} Valid: {valid:<5} Meta: {meta}")


