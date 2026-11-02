"""Command-line chat interface."""
from chatbot import FAQChatbot


def main():
    bot = FAQChatbot()
    print("FAQ Chatbot (type 'quit' to exit)")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            print("Bot: Goodbye!")
            break
        print("Bot:", bot.get_answer(question))


if __name__ == "__main__":
    main()
