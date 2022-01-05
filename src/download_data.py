import json
import requests
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", type=str, default='at',
                        help='dataset selection. at=aide territoire aides. mt=mission transition aides')
    args = parser.parse_args()
    all_data = []
    if args.dataset == "at":
        first_link = "https://aides-territoires.beta.gouv.fr/api/aids/"
    else:
        first_link = "https://aides-territoires.beta.gouv.fr/api/aids/?categories=economie-circulaire&categories=circuits-courts-filieres&categories=economie-denergie&categories=recyclage-valorisation&categories=empreinte-carbone&categories=assainissement&categories=reseaux-de-chaleur&categories=limiter-les-deplacements-subis&categories=mobilite-partagee&categories=mobilite-pour-tous&categories=amenagement-de-lespace-public-et-modes-actifs&categories=transition-energetique&categories=biodiversite&categories=forets&categories=milieux-humides&categories=montagne&categories=qualite-de-lair&categories=risques-naturels&categories=sols&targeted_audiences=private_sector"
    response = requests.get(first_link, verify=False)
    json_file = response.json()
    all_data += json_file["results"]
    while json_file["next"] is not None:
        link = json_file["next"]
        response = requests.get(link, verify=False)
        json_file = response.json()
        all_data += json_file["results"]
    if not os.path.isdir("data"):
        os.makedirs("data")
    file_name = "data/AT_aides_full.json" if args.dataset == "at" else "data/MT_aides.json"
    with open(file_name, 'w') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)

