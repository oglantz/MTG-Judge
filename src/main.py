"""
Main entry point for the application.
"""

import argparse
from query_processer import QueryProcessor
from llm import LLMClient
from slop import get_query_context
from query_tagger import QueryTagger
from slop import build_or_load_index, DB_SOURCE


# def main():
#     parser = argparse.ArgumentParser(description="MTG Judge - Magic: The Gathering rules assistant")
#     parser.add_argument("-q", "--query", required=True, help="Should be followed by a MTG rules question. CARD NAMES MUST BE WRAPPED IN [[name]]")
#     args = parser.parse_args()

#     # Extract the query from the arguments
#     query = args.query
#     query_processor = QueryProcessor()
#     query_context = query_processor.extract_context(query)

#     # Extract rules
#     relevant_rules = get_rules(query_context["cleaned_query"])
#     # Add rules to context
#     query_context["rules_context"] = relevant_rules

#     # print(query_context)
#     llm_client = LLMClient()
#     response = llm_client.generate(query_context)
#     print("----------------------\n\n\n\n LLM RESPONSE: \n -------------")
#     print(response)


def main():
    # print("loading DB")
    # build_or_load_index(DB_SOURCE)



    
    print("Loading models (one-time)...")
    llm_client = LLMClient()
    llm_client._load()  # force load now instead of on first generate()
    
    
    while True:
        query = input("\nAsk a rules question (or 'quit'): ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        
        # query_tagger = QueryTagger()
        # query_processor = QueryProcessor()
        # query_context = query_processor.extract_context(query)
        # tags = query_tagger.tag(query_context["cleaned_query"], query_context["oracle_context"])
        # print(tags)
        query_context = get_query_context(query)
    
        # starting inference...
        print("Starting inference...")
        response = llm_client.generate(query_context)
        print("\n\n--- RULING ---")
        print(response)


if __name__ == "__main__":
    main()