
import os, json, ace_tools_open
import pandas as pd
from Wikipedia_DataWrangling import Manual_WikiSearch, transformer_similarities

def main():


    path = os.path.join(os.getcwd(), 'Data', 'Dictionary', 'synonyms_dict.json')

    # Cargar el diccionario desde el archivo JSON
    with open(path, "r", encoding="utf-8") as archivo:
        synonyms = json.load(archivo)

    print('Size synonyms: ', len(synonyms.keys()))    

    # Gathering concepts considered as the ground truth
    path = os.path.join(os.getcwd(), 'Data', 'Results', 'AIConcepts_Base.csv')
    df = pd.read_csv(path, index_col=False)

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

    pd.DataFrame({'No_Encontrado':noencontrado, "Not_Found":notfound}).to_csv(os.path.join(os.getcwd(), 'Data', 'Wikipedia', "Bad_WikiConcepts_Manual.csv"), index=False)
    df_wiki1 = df_wiki1.loc[idxs2keep]  
    df_wiki1.reset_index(drop=True, inplace=True)
    df_wiki1.to_csv(os.path.join(os.getcwd(), 'Data', 'Wikipedia', "WikiConcepts_Manual.csv"), index=False)
    

    ace_tools_open.display_dataframe_to_user(name="Sample Wiki Data", dataframe=df_wiki1)
    print('Size synonyms: ', len(synonyms.keys()), len(notfound), len(noencontrado))


    # Saving synonims dictitonary
    path2save = os.path.join(os.getcwd(), 'Data', 'Dictionary', 'synonyms_dict.json')
    with open(path2save, "w", encoding="utf-8") as json_file:
        json.dump(synonyms, json_file, indent=4, ensure_ascii=False)  # Pretty formatting with indent=4   
    
    # Webscraping the notfound concepts definittionns usinng the SerpApi service
    #df_wiki2a, _ = SerpAPI_WikiSearch(notfound, " modelo estadistica")
    #df_wiki2b, _ = SerpAPI_WikiSearch(notfound, " machine learning")
    #df_wiki2a.to_csv(os.path.join(os.getcwd(), 'Data', 'Wikipedia', "WikiConcepts_SerpAPI1.csv"), index=False)
    #df_wiki2b.to_csv(os.path.join(os.getcwd(), 'Data', 'Wikipedia', "WikiConcepts_SerpAPI2.csv"), index=False)
    


if __name__ == '__main__':
    main()

