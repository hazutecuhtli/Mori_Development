'''************************************************************************
Importing Libraries
************************************************************************'''
import os, json, sys, ace_tools_open
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Add parent directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from Notebooks.Data_Generation.Wikipedia_DataWrangling import Manual_WikiSearch, transformer_similarities, SerpAPI_WikiSearch
'''************************************************************************
Setting up variables
************************************************************************'''
# Find the .env file
env_path = find_dotenv()
print(".env file found at:", env_path)
# Load it
load_dotenv(env_path)
# Get the key
serpapi_key = os.getenv("SERPAPI_KEY")
print("SERPAPI_KEY =", 'XxXxXxXxXxXxXxXxXxXx')

# If not found, ask the user
if not serpapi_key:
    print("No SerpAPI key found.")
    user_input = input("Do you want to enter it manually now? (y/n): ").strip().lower()
    
    if user_input == "y":
        serpapi_key = input("Please enter your SerpAPI key: ").strip()
        print(f"You can save your key in the {os.path.join(sys.path[0], '.env')} file to be loaded automatically :)")        
    else:
        print("!!Skipping API-dependent part of the script!!")
        print(f"You can save your key in the {os.path.join(sys.path[0], '.env')} file to be loaded automatically :)")
        serpapi_key = None

# Conditional logic
if serpapi_key:
    print("\nAPI key received. Proceeding with SerpAPI calls...")
    # Call your API logic here
else:
    print("No API key provided. API logic skipped.")

'''************************************************************************
Functions
************************************************************************'''
def main():


    path = os.path.join(sys.path[0], 'Notebooks','Data_Generation', 'Data', 'Dictionary', 'synonyms_dict.json')

    # Cargar el diccionario desde el archivo JSON
    with open(path, "r", encoding="utf-8") as archivo:
        synonyms = json.load(archivo)

    print("\nBe Aware!!! Running this code will require a considerable amount of time!\n")

    # Gathering concepts considered as the ground truth
    path = os.path.join(sys.path[0], 'Notebooks','Data_Generation', 'Data', 'Results', 'AIConcepts_Base.csv')
    df = pd.read_csv(path, index_col=False)
    df = df.sample(5)

    # Webscraping the wikipedia website searching for complementary definittions
    df_wiki1, notfound, noencontrado, synonyms = Manual_WikiSearch(df.concept.tolist(), synonyms, language = "en", languageSite = "en")
    df_tmp, notfound, noencontrado, synonyms = Manual_WikiSearch(noencontrado, synonyms, language = "es", languageSite = "es")
    df_wiki1 = pd.concat([df_wiki1, df_tmp], ignore_index=True, sort=False)
    df_tmp, notfound, noencontrado, synonyms = Manual_WikiSearch(noencontrado, synonyms, language = "es", languageSite = "en")
    df_wiki1 = pd.concat([df_wiki1, df_tmp], ignore_index=True, sort=False)
    df_tmp, notfound, noencontrado, synonyms = Manual_WikiSearch(notfound, synonyms, language = "en", languageSite = "es")
    df_wiki1 = pd.concat([df_wiki1, df_tmp], ignore_index=True, sort=False)


    idxs2keep, idxs2rem, sims, bad_sims = transformer_similarities(df, df_wiki1, thre=.3)
    
    for concept, concepto in zip(df_wiki1.loc[idxs2rem].concept, df_wiki1.loc[idxs2rem].concepto):
        notfound.append(concept)
        noencontrado.append(concepto)

    pd.DataFrame({'No_Encontrado':noencontrado, "Not_Found":notfound}).to_csv(os.path.join(sys.path[0], 'Notebooks','Data_Generation', 'Data', 'Wikipedia', "Bad_WikiConcepts_Manual.csv"), index=False)
    df_wiki1 = df_wiki1.loc[idxs2keep]  
    df_wiki1.reset_index(drop=True, inplace=True)
    print('Saving found wikipedia concepts in: ', os.path.join(sys.path[0], 'Notebooks','Data_Generation', 'Data', 'Wikipedia'))
    df_wiki1.to_csv(os.path.join(sys.path[0], 'Notebooks','Data_Generation', 'Data', 'Wikipedia', "WikiConcepts_Manual.csv"), index=False)
    

    #ace_tools_open.display_dataframe_to_user(name="Sample Wiki Data", dataframe=df_wiki1)

    # Saving synonims dictitonary
    path2save = os.path.join(sys.path[0], 'Notebooks','Data_Generation', 'Data', 'Dictionary', 'synonyms_dict.json')
    print('Saving updated dictinary of concepts: ', path2save)
    with open(path2save, "w", encoding="utf-8") as json_file:
        json.dump(synonyms, json_file, indent=4, ensure_ascii=False)  # Pretty formatting with indent=4   
    
    # Webscraping the notfound concepts definittionns usinng the SerpApi service
    if serpapi_key:
        df_wiki2a, _ = SerpAPI_WikiSearch(notfound, " modelo estadistica", serpapi_key)
        df_wiki2b, _ = SerpAPI_WikiSearch(notfound, " machine learning", serpapi_key)
        print('Saving SerpAPI found wikipedia concepts in: ', os.path.join(sys.path[0], 'Notebooks','Data_Generation', 'Data', 'Wikipedia'))        
        df_wiki2a.to_csv(os.path.join(sys.path[0], 'Notebooks','Data_Generation', 'Data', 'Wikipedia', "WikiConcepts_SerpAPI1.csv"), index=False)
        df_wiki2b.to_csv(os.path.join(sys.path[0], 'Notebooks','Data_Generation', 'Data', 'Wikipedia', "WikiConcepts_SerpAPI2.csv"), index=False)
    
    print(f"There were found {df_wiki1.shape[0]+df_wiki2a.shape[0]+df_wiki2b.shape[0]} of {df.shape[0]} conpcets, {df_wiki1.shape[0]} direcly from Wikipedia, and {df_wiki2a.shape[0]+df_wiki2b.shape[0]} using SerpAPI")
    print("Those found concepts and definitions need to be filtered out to remove unrelated topics")


if __name__ == '__main__':
    main()

'''************************************************************************
FIN
************************************************************************'''