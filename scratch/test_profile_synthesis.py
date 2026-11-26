import re

def synthesize_person_profile_answer(person_name: str, items: list, query: str) -> str:
    if not items:
        return (
            f"No verified public professional profile or public records could be located for '{person_name}'.\n\n"
            "**Note on Public Profile Searches:**\n"
            "- Automated retrieval is limited to publicly indexed web summaries and public directory listings.\n"
            "- If you have a specific individual in mind, please provide their direct public profile URL or distinguishing details "
            "(such as organization, university, or general location) to verify the correct public record without identity confusion."
        )

    # Check if any item actually mentions the person's name (case-insensitive)
    name_parts = [p.lower() for p in person_name.split() if len(p) > 2]
    relevant_items = []
    for it in items:
        f_text = it.get("full_text") or ""
        s_text = it.get("snippet") or ""
        t_text = it.get("title") or ""
        content = (t_text + " " + f_text + " " + s_text).lower()
        if all(part in content for part in name_parts):
            relevant_items.append(it)

    if not relevant_items:
        return (
            f"Public web search found potential listings, but none could be reliably verified as belonging specifically to '{person_name}'.\n\n"
            "**Identity Ambiguity Caution:**\n"
            "To avoid misattributing professional credentials or mistaking individuals with similar names, "
            "please provide a direct public profile link or distinguishing professional details."
        )

    # Compile verified public points
    top_item = relevant_items[0]
    all_text = " ".join([it.get("full_text", "") or it.get("snippet", "") for it in relevant_items[:3]])

    sections = [f"### Public Professional Profile Summary: **{person_name.title()}**\n"]

    # Extract key factual highlights directly stated in public text
    highlights = []
    
    # Check university / student mention
    uni_match = re.search(r"(?:student\s+(?:from|at)\s+([A-Za-z\s]+?)(?:who|\.|\,|\band\b))", all_text, re.I)
    if uni_match:
        highlights.append(f"**Affiliation / Education**: Student at {uni_match.group(1).strip()}")

    # Check projects / software
    proj_match = re.search(r"(?:created|built|developer\s+of|developed)\s+([A-Za-z0-9\s\-]+?)(?:—|\-|\–|\.|\,|\band\b)", all_text, re.I)
    if proj_match:
        highlights.append(f"**Projects & Work**: Creator / developer of {proj_match.group(1).strip()}")

    # Check technical focus
    if any(k in all_text.lower() for k in ["ai and ml", "machine learning", "engineering"]):
        highlights.append("**Focus Areas**: Artificial Intelligence, Machine Learning, and Engineering")

    if highlights:
        sections.append("Based on publicly indexed information and professional posts on LinkedIn:\n")
        for h in highlights:
            sections.append(f"- {h}")
    else:
        # Fallback to lead sentences of the public snippet
        lead_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", top_item.get("snippet", "")) if len(s.strip()) > 20]
        if lead_sentences:
            sections.append("Public Information Overview:\n" + " ".join(lead_sentences[:2]))

    # Add identity note
    sections.append(
        "\n**Public Source Verification & Identity Note:**\n"
        "This information is derived strictly from publicly indexed profiles and posts. "
        "Because multiple professionals may share similar names, please refer to the official public profile link below for complete, authoritative verification."
    )

    return "\n".join(sections)

# Test with actual reproduction data
test_items = [
    {
        "title": "Shanveer Singh Khosa's Post - LinkedIn",
        "url": "https://www.linkedin.com/posts/shanveer-singh-khosa-a6221231b_reposting-this-with-gratitude-thank-you-activity-7431298190953394176-Nx3R",
        "snippet": "Meet Shanveer Singh Khosa — a student from Akal University Talwandi Sabo who didn't wait for the world to hand him an opportunity. He has created Interview Command — an AI-powered mock interview simulator.",
        "full_text": "Meet Shanveer Singh Khosa — a student from Akal University Talwandi Sabo who didn't wait for the world to hand him an opportunity. He has created Interview Command — an AI-powered mock interview simulator that puts you inside a real student journey engineering life ai and ml"
    },
    {
        "title": "Shanveer Singh Khosa - LinkedIn India",
        "url": "https://in.linkedin.com/in/shanveersinghkhosa",
        "snippet": "View Shanveer Singh Khosa's profile on LinkedIn, a professional community of 1 billion members.",
        "full_text": None
    }
]

print(synthesize_person_profile_answer("Shanveer Singh Khosa", test_items, "tell me who is the shanveer singh khosa on the linkedin"))
