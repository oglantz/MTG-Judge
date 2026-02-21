"""
Main entry point for the application.
"""

import argparse
from query_processer import QueryProcessor
from llm import LLMClient


def main():
    parser = argparse.ArgumentParser(description="MTG Judge - Magic: The Gathering rules assistant")
    parser.add_argument("-q", "--query", required=True, help="Should be followed by a MTG rules question. CARD NAMES MUST BE WRAPPED IN [[name]]")
    args = parser.parse_args()

    # Extract the query from the arguments
    query = args.query
    query_processor = QueryProcessor()
    query_context = query_processor.extract_context(query)
    
    llm_client = LLMClient()
    response = llm_client.generate(query_context)
    print(response)



if __name__ == "__main__":
    main()