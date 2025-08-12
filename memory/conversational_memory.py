from collections import deque
from memory.pydantic_model import Context
from memory.knowledge_base import ChromaStore
from memory.llm_summarizer import ConversationSummarizer
from config import (
    BUFFER_SIZE,
    SUMMARY_SIZE,
    CHROMA_DB_DIR,
    N_FACTS_CONTEXT,
)


class ConversationManager:
    def __init__(self):
        self.buffer_deque = deque(maxlen=BUFFER_SIZE)
        self.summary_deque = deque(maxlen=SUMMARY_SIZE)

        # Separate stores for different fact types
        self.experiential_store = ChromaStore(
            collection_name="experiential_facts", persist_directory=CHROMA_DB_DIR
        )
        self.personal_store = ChromaStore(
            collection_name="personal_facts", persist_directory=CHROMA_DB_DIR
        )

        self.summarizer = ConversationSummarizer()
        self._list_of_personal_facts_stored_from_conversation = []
        self._list_of_experiential_facts_stored_from_conversation = []

    def process_turn(self, user: str, ai: str):
        """Add turn to buffer; summarize and store when buffer is full."""
        self.buffer_deque.append((user, ai))
        if len(self.buffer_deque) == self.buffer_deque.maxlen:
            self._summarize_and_store_half()

    def _summarize_and_store_half(self):
        """Summarize half of the buffer and update memory."""
        half_size = len(self.buffer_deque) // 2
        if half_size == 0:
            return

        data_to_summarize = [self.buffer_deque.popleft() for _ in range(half_size)]
        summary_ctx = self.summarizer.summarize_conversation(data_to_summarize)
        with open("conversation_summary.txt", "a", encoding="utf-8") as f:
            f.write(summary_ctx.summary + "\n")
        self._handle_summary(summary_ctx)

    def _handle_summary(self, ctx: Context):
        """Store summary in deque and save important facts to Chroma."""
        self.summary_deque.appendleft(ctx)

        if len(self.summary_deque) == self.summary_deque.maxlen:
            old_ctx = self.summary_deque.pop()
            self._store_facts_if_worthy(old_ctx)

    def _store_facts_if_worthy(self, ctx: Context):
        """Persist facts if marked important."""
        if ctx and ctx.summary != "Error occurred during summarization":
            if ctx.experiential_facts:
                for fact in ctx.experiential_facts:
                    if fact and fact.strip():
                        self.experiential_store.add_document(fact)
                        self._list_of_experiential_facts_stored_from_conversation.append(
                            fact
                        )
            if ctx.personal_facts:
                for fact in ctx.personal_facts:
                    if fact and fact.strip():
                        self.personal_store.add_document(fact)
                        self._list_of_personal_facts_stored_from_conversation.append(
                            fact
                        )

    def get_context_for_ai(self, user_query: str) -> str:
        """Build AI context from buffer, summaries, and relevant long-term memory."""
        recent_conv = "\n".join(
            f"User: {u}\nAI: {a}" for u, a in list(self.buffer_deque)[-7:]
        )

        summary_memory = "\n".join(s.summary for s in self.summary_deque)

        experiential_info = self.experiential_store.get_relevant_info(
            user_query, n_results=N_FACTS_CONTEXT
        )
        personal_info = self.personal_store.get_relevant_info(
            user_query, n_results=N_FACTS_CONTEXT
        )

        context = f"""
--- Current Conversation ---
{recent_conv}
            
--- Past Conversation Summary ---
{summary_memory}
            
--- Relevant Long-Term Experiential Facts ---
{experiential_info}

--- Relevant Long-Term Personal Facts ---
{personal_info}
"""

        with open("context.txt", "a", encoding="utf-8") as f:
            f.write(context + "\n\n")

        return context

    def end_conversation(self):
        """Summarize and store all remaining conversation, then clear memory."""
        for ctx in list(self.summary_deque):
            self._store_facts_if_worthy(ctx)

        if self.buffer_deque:
            final_summary = self.summarizer.summarize_conversation(
                list(self.buffer_deque)
            )
            self._store_facts_if_worthy(final_summary)

        self.buffer_deque.clear()
        self.summary_deque.clear()
        with open("stored_facts.txt", "a", encoding="utf-8") as f:
            f.write("experiential_facts:\n")
            f.write(
                "\n".join(self._list_of_experiential_facts_stored_from_conversation)
                + "\n"
            )
            f.write("personal_facts:\n")
            f.write(
                "\n".join(self._list_of_personal_facts_stored_from_conversation) + "\n"
            )
            f.write("Conversation ended and stored successfully.\n")
        return "Conversation ended and stored successfully."
