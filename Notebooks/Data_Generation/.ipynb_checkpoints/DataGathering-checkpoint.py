
import os, glob, json
import pandas as pd
from tqdm import tqdm
from Wikipedia_DataWrangling import Text1stsentence, translation_function
from DataWrangling_Functions import GeneratingData_WebScraping, GatheringData_PDFReader1
from DataWrangling_Functions import GatheringData_PDFReader2, GatheringData_PDFReader3
from DataWrangling_Functions import GatheringData_PDFReader4
from DataCleaning_Functions import remove_parenthesis_complements

def main():
    
    # Fetch the web page
    url1 = 'https://developers.google.com/machine-learning/glossary'#"https://developers.google.com/machine-learning/glossary?hl=es-419"
    url2 = "https://iaarbook.github.io/glosario/"
    #url3 - 'https://www.datacamp.com/es/blog/data-science-glossary'
    urls = [url1, url2]
    headers_types = ['h2', 'h3']
    clases = ['hide-from-toc', None]
    text_ids = ['data-text', 'id']
    filenames = ['MLConcepts_Google.csv', 'MLConcepts_Iaabook.csv']
    paragraphs = 2

    df_web, synonyms = GeneratingData_WebScraping(urls, headers_types , clases, text_ids, paragraphs)
    df_web.loc[:, "concept"] = df_web.concept.str.strip().str.lower().str.replace("_", " ").str.replace("-", " ").str.replace("  ", " ").str.replace(",", "")

    path2save = os.path.join(os.getcwd(), 'Data', 'Results', 'Webscraping_Concepts.csv')
    df_web.to_csv(path2save, index=False)

    
    print(df_web.loc[20:40].head(20))


    books = ['Glosario_MOOC_Big_Data.pdf',
             'libro_01_ciencia_de_datos_teoria_y_aplicaciones.pdf',
             'BigDataConcepts.pdf',
             'LibroEstadistica.pdf']

    paths = []
    for book in books:
        paths.append(os.path.join(os.getcwd(), 'Books', book))

    print(paths[0])
    df_pdf = GatheringData_PDFReader1(paths[0])
    df_pdf.to_csv(os.path.join(os.getcwd(), 'Data', 'Books', books[0].split('.')[0]+'.csv'))
    print(paths[1])
    df_tmp = GatheringData_PDFReader2(paths[1])
    df_tmp.to_csv(os.path.join(os.getcwd(), 'Data', 'Books', books[1].split('.')[0]+'.csv'))
    df_pdf = pd.concat([df_pdf, df_tmp], ignore_index=True, sort=False)
    print(paths[2])
    df_tmp = GatheringData_PDFReader3(paths[2])
    df_tmp.to_csv(os.path.join(os.getcwd(), 'Data', 'Books', books[2].split('.')[0]+'.csv'))
    df_pdf = pd.concat([df_pdf, df_tmp], ignore_index=True, sort=False)
    print(paths[3])
    df_tmp = GatheringData_PDFReader4(paths[3])
    df_tmp.to_csv(os.path.join(os.getcwd(), 'Data', 'Books', books[3].split('.')[0]+'.csv'))
    df_pdf = pd.concat([df_pdf, df_tmp], ignore_index=True, sort=False)    


    path2save = os.path.join(os.getcwd(), 'Data', 'Results', 'AIBooks_Concepts.csv')
    #df = pd.concat([df_web, df_pdf], ignore_index=True, sort=False)
    df_pdf.loc[:, "concepts"] = df_pdf.concepts.str.strip().str.lower().str.replace("-", " ").str.replace("_", " ").str.replace("“", "").str.replace("”", "").str.replace("  ", " ").str.replace(",", "")
    df_pdf.loc[:, "concepts"] = df_pdf.concepts.apply(lambda x: remove_parenthesis_complements(x))
    df_pdf.loc[:, "concepts"] = df_pdf.concepts.str.replace("ndice", "índice").str.replace("obre", "sobre")
    df_pdf.loc[:, "definitions"]  = df_pdf.definitions.str.strip().str.replace("“", '"').str.replace("”", '"').str.replace("  ", " ").str.capitalize()
    #df.loc[:, "definitions"] = df.definitions.apply(lambda x: remove_parenthesis_complements(x))
    df_pdf.rename(columns={"concepts":"concepto", 'definitions':'descripcion'}, inplace=True)
    df_pdf.loc[:, 'concept'] = df_pdf.concepto.map({v: k for k, v in synonyms.items()})

    print()
    for idx in tqdm(df_pdf.loc[df_pdf.index[df_pdf.concept.isna()]].index, mininterval=50, total=df_pdf.loc[df_pdf.index[df_pdf.concept.isna()]].index.shape[0], desc="Processing", unit="row"):                              
        df_pdf.loc[idx, 'concept'] = translation_function(df_pdf.loc[idx, 'concepto'], english=True)

    for concept, concepto in zip(df_pdf.concept, df_pdf.concepto):
        synonyms[concept.lower()] = concepto.lower()

    df_pdf.loc[:, "concept"] = df_pdf.concept.str.strip().str.lower().str.replace("_", " ").str.replace("-", " ").str.replace("  ", " ").str.replace(",", "")
    df_pdf.to_csv(path2save, index=False)
    print(df_pdf.loc[20:40].head(20))


    path = os.path.join(os.getcwd(), 'Data', 'GPT')
    files = glob.glob(os.path.join(path, '*.csv'))

    for i, file in enumerate(files):
        if i == 0:
            df_GPT = pd.read_csv(file, index_col=False)
            df_GPT.loc[:, 'Source'] = 'GPT'
        else:
            df_tmp = pd.read_csv(file, index_col=False)
            df_tmp.loc[:, 'Source'] = 'GPT'
            df_GPT = pd.concat([df_GPT, df_tmp], ignore_index=True, sort=False)

    print("\n")
    path2save = os.path.join(os.getcwd(), 'Data', 'Results', 'GPT_Concepts.csv')

    df_GPT.loc[:, "Concepto"] = df_GPT.Concepto.str.strip().str.lower().str.replace("-", " ").str.replace("_", " ").str.replace("“", "").str.replace("”", "").str.replace("  ", " ").str.replace(",", "")
    df_GPT.loc[:, "Concepto"] = df_GPT.Concepto.apply(lambda x: remove_parenthesis_complements(x))
    df_GPT.loc[:, "Descripción"]  = df_GPT["Descripción"].str.strip().str.replace("“", '"').str.replace("”", '"').str.replace("  ", " ").str.capitalize()

    df_GPT.rename(columns={"Concepto":"concepto", 'Descripción':'descripcion'}, inplace=True)
    df_GPT.loc[:, 'concept'] = df_GPT.concepto.map({v: k for k, v in synonyms.items()})

    for idx in tqdm(df_GPT.loc[df_GPT.index[df_GPT.concept.isna()]].index, mininterval=50, total=df_GPT.loc[df_GPT.index[df_GPT.concept.isna()]].index.shape[0], desc="Processing", unit="row"):
        df_GPT.loc[idx, 'concept'] = translation_function(df_GPT.loc[idx, 'concepto'], english=True)

    df_GPT.loc[:, 'concept'] = df_GPT.concept.str.lower()    

    for concept, concepto in zip(df_GPT.concept, df_GPT.concepto):
        synonyms[concept.lower()] = concepto.lower()
        
    #df_GPT.loc[:, "Descripción"] = df_GPT["Descripción"].apply(lambda x: remove_parenthesis_complements(x))
    df_GPT.to_csv(path2save, index=False)
    print(df_GPT.loc[20:40].head(20))


    # Saving synonims dictitonary
    path2save = os.path.join(os.getcwd(), 'Data', 'Dictionary', 'synonyms_dict.json')
    with open(path2save, "w", encoding="utf-8") as json_file:
        json.dump(synonyms, json_file, indent=4, ensure_ascii=False)  # Pretty formatting with indent=4    


if __name__ == '__main__':
    main()

