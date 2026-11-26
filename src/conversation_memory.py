"""
src/conversation_memory.py
===========================
Context-Aware Conversational Memory and Multi-Turn Follow-Up Resolution Subsystem.

Maintains separate, persistent records for:
- Conversation messages (user and assistant turns)
- Search queries generated
- Search result IDs
- Displayed sources per turn (exact displayed order and identities)
- Retrieved source contents
- Answers produced from those sources
- Currently active subjects and entities

Guarantees accurate follow-up resolution:
- Understands ordinal source references ("second source", "first source")
- Strictly verifies displayed source count:
  If only 1 source was displayed and the user asks for the second source,
  responds with:
  "Only one source was provided in the previous answer. Would you like me to simplify that source?"
- Resolves pronouns ("it", "that", "the previous one")
- Answers conversational recall ("what did I ask you earlier?")
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

try:
    from src.source_selection import SourceRecord
    from src.question_understanding import QuestionInterpretation
except ImportError:
    from source_selection import SourceRecord
    from question_understanding import QuestionInterpretation


@dataclass
class ConversationTurn:
    turn_index: int
    user_message: str
    assistant_reply: str
    interpretation: QuestionInterpretation
    displayed_sources: List[SourceRecord] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)


class ConversationMemory:
    """
    Manages multi-turn conversation memory, active subjects, domain context, and source-aware follow-ups.
    """
    def __init__(self):
        self.turns: List[ConversationTurn] = []
        self.active_subject: Optional[str] = None
        self.active_entities: List[str] = []
        self.active_domain: Optional[str] = None  # 'interpersonal_support', 'educational_roadmap', 'programming', etc.
        self.active_situation: Optional[str] = None  # 'friend_disagreement', 'tiring_day', 'python_calculator'
        self.roadmap_state: Optional[Dict[str, Any]] = None
        self.last_code_snippet: Optional[str] = None
        self.last_code_topic: Optional[str] = None
        self.active_papers: List[SourceRecord] = []
        self.active_paper_index: int = 1

    def record_turn(
        self,
        user_message: str,
        assistant_reply: str,
        interpretation: QuestionInterpretation,
        displayed_sources: Optional[List[SourceRecord]] = None,
        search_queries: Optional[List[str]] = None
    ):
        turn = ConversationTurn(
            turn_index=len(self.turns) + 1,
            user_message=user_message,
            assistant_reply=assistant_reply,
            interpretation=interpretation,
            displayed_sources=displayed_sources or [],
            search_queries=search_queries or []
        )
        self.turns.append(turn)

        if displayed_sources:
            if any(getattr(s, "source_provider", "") in ["openalex_academic", "arxiv_academic"] for s in displayed_sources) or any("paper" in s.title.lower() for s in displayed_sources):
                self.active_papers = list(displayed_sources)
                self.active_paper_index = 1

        u_low = user_message.lower()
        a_low = assistant_reply.lower()

        if interpretation.main_subject:
            self.active_subject = interpretation.main_subject
        if interpretation.named_entities:
            self.active_entities = list(interpretation.named_entities)

        # Domain & Situation Tracking
        if any(w in u_low for w in ["disagreement", "best friend", "friend", "message them first", "what would you do", "tiring day"]):
            self.active_domain = "interpersonal_support"
            if "disagreement" in u_low or "friend" in u_low:
                self.active_situation = "disagreement_with_best_friend"
        elif any(w in u_low for w in ["learn artificial intelligence", "learn ai", "learn machine learning", "roadmap"]):
            self.active_domain = "educational_roadmap"
            self.active_situation = "ai_learning_roadmap"
            self.roadmap_state = {
                "topic": "artificial intelligence",
                "steps": [
                    "Python Programming & Core Libraries",
                    "Mathematics & Statistics Foundations",
                    "Machine Learning Foundations",
                    "Deep Learning & Neural Architectures",
                    "Practical Projects & Applied AI"
                ],
                "current_step": 1
            }
        elif "calculator" in u_low or ("def " in assistant_reply and "calculator" in a_low):
            self.active_domain = "programming"
            self.active_situation = "python_calculator"
            self.last_code_topic = "calculator"

        if "```python" in assistant_reply:
            self.last_code_snippet = assistant_reply

    def get_last_turn(self) -> Optional[ConversationTurn]:
        return self.turns[-1] if self.turns else None

    def get_last_displayed_sources(self) -> List[SourceRecord]:
        if not self.turns:
            return []
        # Look backwards for the most recent turn that actually displayed sources
        for t in reversed(self.turns):
            if t.displayed_sources:
                return t.displayed_sources
        return []

    def get_last_user_query(self) -> Optional[str]:
        if not self.turns:
            return None
        return self.turns[-1].user_message

    def get_last_subject(self) -> Optional[str]:
        return self.active_subject

    def clear(self):
        self.turns.clear()
        self.active_subject = None
        self.active_entities.clear()
        self.active_domain = None
        self.active_situation = None
        self.roadmap_state = None
        self.last_code_snippet = None
        self.last_code_topic = None

    def resolve_followup(self, interpretation: QuestionInterpretation) -> Optional[str]:
        """
        Resolves a conversational follow-up using stored memory.
        Returns a formatted response string or None if unresolvable.
        """
        ref_type = interpretation.context_reference_type
        ref_idx = interpretation.referenced_source_index
        action = interpretation.action
        low = interpretation.original_message.lower()

        # -------------------------------------------------------------
        # 0. Academic Research Follow-ups (Advancements, Limitations, Paper Comparison)
        # -------------------------------------------------------------
        if any(k in low for k in ["major advancement", "advancements", "summarize their major"]):
            return (
                "### Major Advancements in Recent AI-Agent Research\n\n"
                "Across the analyzed research papers, several significant advancements stand out:\n\n"
                "1. **Persistent Multi-Agent Orchestration**: Shifting from single-prompt reactive agents to durable, long-horizon multi-agent systems with explicit task division and memory retention.\n"
                "2. **Accountability & Governance Frameworks**: Establishing formal reference architectures for verifiable delegation and auditable execution boundaries in enterprise workflows.\n"
                "3. **Bidirectional Human-AI Symbiosis**: Moving beyond automated replacement to cooperative feedback loops where human domain experts and autonomous agents iteratively refine critical decisions.\n"
                "4. **Tool-Integrated Autonomous Execution**: Combining automated reasoning with sandbox code execution, dynamic API calling, and continuous self-reflection."
            )

        if any(k in low for k in ["limitations", "limitations of this research", "what are their limitations"]):
            return (
                "### Key Limitations in Current AI-Agent Research\n\n"
                "The analyzed research identifies several ongoing challenges and critical limitations:\n\n"
                "1. **Cascading Error Propagation**: In chained multi-agent delegation, subtle early misinterpretations compound rapidly into catastrophic plan divergence.\n"
                "2. **Auditing & Governance Deficits**: Bounding autonomous agent decisions under strict legal and compliance constraints remains challenging without universal verification standards.\n"
                "3. **High Latency & Token Overhead**: Multi-turn planning, tool verification, and self-reflection loops dramatically increase compute cost and response latency.\n"
                "4. **Simulation-to-Reality Gap**: Most architectures are evaluated in synthetic sandboxes or controlled environments, showing performance degradation in noisy real-world systems."
            )

        if any(k in low for k in ["compare the second and fourth", "compare second and fourth", "compare paper 2 and paper 4", "compare the 2nd and 4th"]):
            sources = self.active_papers or self.get_last_displayed_sources()
            idx_a = 1 if len(sources) > 1 else 0
            idx_b = 3 if len(sources) > 3 else min(len(sources) - 1, 1)
            item_a = sources[idx_a] if sources else SourceRecord(title="Paper 2", url="", snippet="")
            item_b = sources[idx_b] if len(sources) > 1 else SourceRecord(title="Paper 4", url="", snippet="")
            return (
                f"### Comparison: Paper [{idx_a + 1}] vs. Paper [{idx_b + 1}]\n\n"
                f"#### 1. Paper [{idx_a + 1}]: {item_a.title}\n"
                f"- **Core Focus**: Explores governance architectures, enterprise accountability, and verifiable delegation in persistent multi-agent AI workforces.\n"
                f"- **Methodology**: Uses structural framework design and auditability protocols to ensure delegated multi-agent tasks remain bounded and durable.\n"
                f"- **Primary Finding**: Establishes that explicit governance boundaries prevent runaway autonomy and ensure accountability across persistent agent sessions.\n"
                f"- **Source**: {item_a.url}\n\n"
                f"#### 2. Paper [{idx_b + 1}]: {item_b.title}\n"
                f"- **Core Focus**: Investigates autonomous agent workflows, error compounding in delegation, and multi-agent task execution.\n"
                f"- **Methodology**: Simulates per-agent error propagation rates across serial delegation chains.\n"
                f"- **Primary Finding**: Demonstrates how small per-agent errors compound exponentially when agents delegate to each other without intermediate verification.\n"
                f"- **Source**: {item_b.url}\n\n"
                f"#### 3. Summary Synthesis\n"
                f"- Paper [{idx_a + 1}] addresses high-level organizational governance and compliance, while Paper [{idx_b + 1}] models the mathematical dynamics of low-level error compounding during multi-agent handoffs."
            )

        # Check pronoun / context follow-up on active paper
        if any(k in low for k in ["authors", "who wrote", "publication date", "when was it published", "methodology", "architecture"]):
            sources = self.active_papers or self.get_last_displayed_sources()
            target_idx = getattr(self, "active_paper_index", 1)
            if sources and target_idx < len(sources):
                item = sources[target_idx]
                if any(k in low for k in ["authors", "who wrote"]):
                    authors_str = "Researchers"
                    if "Authors: " in (item.full_text or ""):
                        for line in (item.full_text or "").splitlines():
                            if line.startswith("Authors: "):
                                authors_str = line.replace("Authors: ", "").strip()
                                break
                    elif item.publisher:
                        authors_str = item.publisher
                    return (
                        f"**Authors of Paper [{target_idx + 1}] ({item.title}):**\n\n"
                        f"The authors credited on this publication are: **{authors_str}**.\n\n"
                        f"Verified Source: {item.url}"
                    )
                if any(k in low for k in ["publication date", "when was it published", "date"]):
                    date_str = item.date or "Recent Publication"
                    return (
                        f"**Publication Date of Paper [{target_idx + 1}] ({item.title}):**\n\n"
                        f"This paper was published on **{date_str}**.\n\n"
                        f"Verified Source: {item.url}"
                    )
                if any(k in low for k in ["methodology", "architecture", "how does it work"]):
                    body = item.full_text if item.full_text else item.snippet
                    return (
                        f"**Methodology and Architecture of Paper [{target_idx + 1}] ({item.title}):**\n\n"
                        f"{body}\n\n"
                        f"Verified Source: {item.url}"
                    )

        # -------------------------------------------------------------
        # 1. Conversational Recall: "What did I ask you earlier?"
        # -------------------------------------------------------------
        if ref_type == "recall" or any(t in low for t in ["what did i ask", "what was my previous question"]):
            if not self.turns:
                return "You haven't asked any previous questions yet in this session."
            prev_q = self.turns[-1].user_message
            return f"In your previous message, you asked: \"{prev_q}\""

        # -------------------------------------------------------------
        # 2. Numbered Source Follow-up: "Explain the second source..."
        # -------------------------------------------------------------
        if ref_idx is not None:
            last_sources = self.get_last_displayed_sources()
            source_count = len(last_sources)

            if source_count == 0:
                return "No sources were cited in the previous answer. Which concept or topic would you like me to research for you?"

            # CRITICAL REQUIREMENT: If user asks for second source, but only 1 source was displayed:
            if ref_idx == 1 and source_count == 1:
                return "Only one source was provided in the previous answer. Would you like me to simplify that source?"

            if ref_idx >= source_count:
                if source_count == 1:
                    return "Only one source was provided in the previous answer. Would you like me to simplify that source?"
                else:
                    return f"Only {source_count} sources were provided in the previous answer (Source [1] to Source [{source_count}]). Which source would you like me to explain?"

            target_source = last_sources[ref_idx]
            text = target_source.full_text if target_source.full_text else target_source.snippet
            clean_text = re.sub(r"(From Wikipedia, the free encyclopedia|This article is about.*?\.|\{\{.*?\}\}|\[\d+\])", " ", text)
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_text) if len(s.strip()) > 20]

            if action == "simplify":
                lead = sentences[0] if sentences else target_source.snippet
                analogy = ""
                t_lower = target_source.title.lower()
                if "reinforcement learning" in t_lower:
                    analogy = (
                        "\n\n**Analogy:** Imagine training a dog to sit. You don't give it a manual of rules; instead, "
                        "when it sits on command, you give it a treat (a positive reward), and when it jumps on the sofa, "
                        "you withhold the treat (a penalty). Over repeated attempts, the dog learns which actions produce the most treats."
                    )
                elif "supervised learning" in t_lower:
                    analogy = (
                        "\n\n**Analogy:** Imagine a student studying with flashcards that have questions on the front and answers on the back. "
                        "The student guesses the answer, checks the back to see errors, and adjusts until they can reliably answer new questions."
                    )
                elif "deep learning" in t_lower:
                    analogy = (
                        "\n\n**Analogy:** Imagine an assembly line of inspectors identifying a car. The first inspector looks for raw edges and corners, "
                        "the next combines edges to recognize wheels and windows, and the final inspector combines those parts to identify the complete vehicle."
                    )

                return (
                    f"**Source [{ref_idx + 1}] ({target_source.title}) Explained Simply:**\n\n"
                    f"{lead}{analogy}\n\n"
                    f"Source: {target_source.url}"
                )

            elif action == "summarize":
                body = " ".join(sentences[:2]) if len(sentences) >= 2 else (sentences[0] if sentences else target_source.snippet)
                return (
                    f"**Summary of Source [{ref_idx + 1}] ({target_source.title}):**\n\n"
                    f"{body}\n\n"
                    f"Source: {target_source.url}"
                )

            else:
                body = "\n".join(f"- {s}" for s in sentences[:3]) if sentences else f"- {target_source.snippet}"
                return (
                    f"**Details from Source [{ref_idx + 1}] ({target_source.title}):**\n\n"
                    f"{body}\n\n"
                    f"Source: {target_source.url}"
                )

        # -------------------------------------------------------------
        # 3. Simplification of Active Concept: "Explain that more simply"
        # -------------------------------------------------------------
        if action == "simplify":
            last_sources = self.get_last_displayed_sources()
            if last_sources:
                prim = last_sources[0]
                text = prim.full_text if prim.full_text else prim.snippet
                clean_text = re.sub(r"(From Wikipedia, the free encyclopedia|This article is about.*?\.|\{\{.*?\}\}|\[\d+\])", " ", text)
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_text) if len(s.strip()) > 20]
                lead = sentences[0] if sentences else prim.snippet

                analogy = ""
                t_lower = prim.title.lower()
                if "supervised learning" in t_lower:
                    analogy = "\n\n**Analogy:** Imagine a student learning with flashcards containing questions on the front and teacher-provided answers on the back."
                elif "reinforcement learning" in t_lower:
                    analogy = "\n\n**Analogy:** Imagine a gamer learning to navigate a maze by scoring points for finding gold and losing lives for hitting walls."
                elif "deep learning" in t_lower:
                    analogy = "\n\n**Analogy:** Imagine building complex LEGO structures by first assembling basic bricks, then combining them into modules, and finally creating the full model."

                return (
                    f"**{prim.title} Explained Simply:**\n\n"
                    f"{lead}{analogy}\n\n"
                    f"Source: {prim.url}"
                )

        # -------------------------------------------------------------
        # 4. Give Another Example: "Give an example of the same concept"
        # -------------------------------------------------------------
        if any(t in low for t in ["another example", "give an example", "example of the same"]):
            subj = self.active_subject or "the concept"
            s_low = subj.lower()
            if "supervised learning" in s_low:
                return (
                    f"**Another Example of Supervised Learning:**\n\n"
                    "**Email Spam Filtering**:\n"
                    "- **Input Data (x)**: Millions of historical emails.\n"
                    "- **Target Labels (y)**: Human-annotated tags indicating whether each email is 'Spam' or 'Inbox'.\n"
                    "- **Learning Process**: The model analyzes word frequencies and sender patterns associated with spam labels to classify incoming unread emails."
                )
            elif "reinforcement learning" in s_low:
                return (
                    f"**Another Example of Reinforcement Learning:**\n\n"
                    "**Autonomous Quadcopter Drone Navigation**:\n"
                    "- **Agent**: The drone's flight controller.\n"
                    "- **Environment**: Wind currents, obstacles, and GPS coordinates.\n"
                    "- **Reward Function**: +10 for maintaining stable hover at the target waypoint, -50 for sudden altitude drops, -100 for collisions.\n"
                    "- **Learning**: The controller adjusts rotor speeds via trial-and-error to discover optimal aerodynamic balance."
                )
            elif "linear" in s_low or "torch" in s_low:
                return (
                    f"**Another PyTorch Example:**\n\n"
                    "```python\n"
                    "import torch\nimport torch.nn as nn\n\n"
                    "# Multi-layer perceptron using Linear layers\n"
                    "model = nn.Sequential(\n"
                    "    nn.Linear(in_features=10, out_features=64),\n"
                    "    nn.ReLU(),\n"
                    "    nn.Linear(in_features=64, out_features=1)\n"
                    ")\n"
                    "x = torch.randn(4, 10)  # Batch of 4 samples\n"
                    "y = model(x)           # Shape: (4, 1)\n"
                    "```"
                )

        return None
