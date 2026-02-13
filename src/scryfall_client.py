import requests
import time
import json

"""
Scryfall API Client

returns card info in the format:
{
    "name": "Card Name",
    "oracle_text": "Card Oracle Text",
    "rulings": [
        {
            "object": "ruling",
            "oracle_id": "Ruling Oracle ID",
            "source_name": "Ruling Source",
            "published_at": "Ruling Published Date",
            "comment": "Ruling Text",
        }
    ],
    "has_rulings": bool
}
"""

class ScryfallClient:

    BASE_URL = "https://api.scryfall.com"

    def __init__(self):
        self.delay = 0.1 # 100ms delay between requests to avoid rate limiting
    
    def _get_card_by_name(self, name: str) -> json:
        """Fetches card data by name"""
        time.sleep(self.delay) # rate limiting
        url = f"{self.BASE_URL}/cards/named?exact={name}"

        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def _get_card_rulings(self, rulings_uri: str) -> json:
        """Fetches card rulings from the given URI"""
        time.sleep(self.delay) # rate limiting
        response = requests.get(rulings_uri)
        response.raise_for_status()
        return response.json()
    
    def get_card_info(self, card_name: str) -> json:
        """Get card info by name (name, oracle_text, rulings)"""
        try:
            # card data
            card = self._get_card_by_name(card_name)
            # oracle text
            oracle_text = card.get("oracle_text", None)
            # rulings
            rulings_uri = card.get("rulings_uri", None)
            rulings = []

            if rulings_uri:
                rulings_data = self._get_card_rulings(rulings_uri)
                rulings = rulings_data.get('data', [])
        

            return {
                "name": card.get('name'),
                "oracle_text": oracle_text,
                "rulings": rulings,
                "has_rulings": True if len(rulings) > 0 else False
            }
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return {"error": "Card not found"}
            raise


def main():
    client = ScryfallClient()
    print(client.get_card_info("Humility"))


if __name__ == "__main__":
    main()