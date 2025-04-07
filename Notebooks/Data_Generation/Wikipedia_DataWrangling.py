import requests, os, re, json, spacy, re, unidecode

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Auto-download if missing
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
import os, random, json, nltk, glob, unidecode, re, warnings, torch, spacy
from tqdm import tqdm
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup, NavigableString, Tag
from http.client import responses
from langdetect import detect
from deep_translator import GoogleTranslator


def clean_sentence_endings(text):
    
    # Regex pattern to match "(letter." at the end of a sentence
    pattern = r'\s*\([A-Za-z]\.\s*$'  # Matches " (a." or " (B." etc.
    
    # Remove the matching part from the text
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text


def remove_parenthesis_complements(text):
    """
    Removes text within parentheses along with the parentheses themselves.
    """
    cleaned_text = re.sub(r"\([^)]*\)", "", text)
    return cleaned_text.strip()


def extract_parentheses_with_brackets(text):
    # Regex pattern to capture content including parentheses
    pattern = r'\([^\)]*\)'  # Matches anything between ( and ), including them
    
    # Find all matches
    matches = re.findall(pattern, text)
    
    return matches


def removepar(txt):

    '''
    Function that filter out text, based on the presence of ()

    input:

    txt -> Text to process

    output:

    txt -> Processed text

    '''
    #pattern = r'\((.*?)\)'
    pattern = r'\([^\)]*\)' 
    paren_content  = re.findall(pattern, txt)
    for content in paren_content:
        if (len(content.split(' ')))>12:
            txt = txt.replace(content, "")
        
    return txt      



def wikisearch(url):

    '''
    Functions for webscraping wikipedia sites

    inputs:

    url -> wikipedia url
    
    outputs:

    content -> Information obtained from the input url

    '''    

    # Defining function local variables o parameters
    clase = "mw-heading mw-heading2"
    header_type = "div"
    response = requests.get(url)
    pattern = r"^.*?[.!?](?=\s|$)"

    # Returning no information if response status is not 200
    if response.status_code != 200:
        return None
    
    # Gathering data
    data = response.text
    # Parse the HTML content (Retriving information directly from the website p headers)
    soup = BeautifulSoup(data, 'html.parser')
    content_total = soup.find_all("p")

    # Formating information for the function output
    if content_total[0]:
        content = content_total[0].text
        expressions2remove = ["\n\n\n\n\n{\\displaystyle", "\n", "\\"]
        for expre in expressions2remove:
            content = content.replace(expre, "")
        if len(content)>40:
            content = content
        elif len(content_total)>1:
            content = content_total[1].text
            for expre in expressions2remove:
                content = content.replace(expre, "")                    
    else:
        return None

    # Retrieving information if not present within the first p headers, using a different header
    if len(content) == 0:   
        content = soup.find_all(header_type, class_="mw-content-ltr mw-parser-output")
        content = content[0].find_all("p")
        if len(content)>0:
            for n in range(len(content)):
                if len(content[n].text)>20:
                    content = content[n].text
                    expressions2remove = ["\n\n\n\n\n{\\displaystyle", "\n", "\\"]
                    for expre in expressions2remove:
                        content = content.replace(expre, "")
                    return content
            return None
        else:
            return None


    if len(content_total)>1:
        # Generating the function output that contains information retrieved from the URL input
        if (("refer" not in content_total[1].text) & (len(content_total[1].text)>90)) | (("refer" not in content_total[0].text) & (len(content_total[0].text)>90)):
            pattern = r"\[\d+\]"
            return re.sub(pattern, "", content)
        else:
            # Retrieving information if a different webpage code structure as the two previously used is present
            try:           
                headers = soup.find_all(header_type, class_=clase)
                for n in range(len(headers)):
                    if ("comput" in headers[n].text.lower()) | ("artifi" in headers[n].text.lower()) | ("stat" in headers[n].text.lower())| ("mat" in headers[n].text.lower()):
                        initial_header = headers[n]
                        if n < len(headers):
                            final_header = headers[n+1]
                for element in initial_header.find_next_siblings():
                    if element == final_header:
                        break
                    else:
                        prev_element = element
            
                content = []
                if prev_element.name in ["ul", "ol"]:
                    list_items = prev_element.find_all("li")
                    for item in list_items:
                        if item.text not in content:
                            content.append(item.text)
                            if item.find_all("ul") is not None:
                                idx = len(content)
                                if idx > 0:
                                    idx = idx-1
                                    content[idx] = item.text.replace("\n", "")
                                else:
                                    idx = 0
                                for item2 in (item.find_all("li")):
                                    content[idx] = content[idx].replace(item2.text.replace("\n", ""), "")
                                    content.append(item2.text)
                            else:
                                content.append(item.text)
                
                if len(content)>0:
                    # Generating the function output that contains information retrieved from the URL input
                    return content
            except:
                return None

    return None



def translation_function(text, english=True):


    '''
    Functions for webscraping wikipedia sites

    inputs:

    text -> text to be translated
    english -> English translation (True) or spanish (False)
    
    outputs:

    translated_text -> Translated text

    '''        

    if english:
        translated_text = GoogleTranslator(source='es', target='en').translate(text)
    else:
        translated_text = GoogleTranslator(source='en', target='es').translate(text)

    return translated_text



def get_wikipedia_link(concept, API_KEY):

    '''
    Function to retrieve Wikipedia links by searching on Google using SerpApi

    inputs:

    concept -> Concept to be searched on google
    API_KEY -> SeraApi API key for webscraping
    
    outputs:

    link -> Translated text

    '''  

    # Defining SerpApi search parameters
    url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": f"{concept} site:wikipedia.org",
        "api_key": API_KEY
    }

    # Implementing the websearch
    response = requests.get(url, params=params)
    data = response.json()

    # Extract first Wikipedia link
    for result in data.get("organic_results", []):
        link = result.get("link", "")
        if "wikipedia.org" in link:
            # Generating the wikipedia link output, if exists
            return link

    #print("No Wikipedia link found.")
    return None



def Text1stsentence(text):


    '''
    Function to extract the first sentence of the retrieved concepts definitions
    
    inputs:

    text -> Retrieved concept definition from the web

    outputs:

    new_txt -> Extracted first sentence from the input text

    '''


    # Fixing words that affect the cleansing processes
    text = text.replace('0.0', '0').replace('1.0', '1')

    pattern = r'\s*\[\d+\]'
    text = re.sub(pattern, '', text)

    text = removepar(text)

    pattern = r'\s*\([^)]*$'
    text = re.sub(pattern, '', text)

    # Fixing words within the retrieved data sentences
    text = removepar(text)

    dict4correct = {"gu stos":"gustos", "l ógica":"lógica", "di álogo":"diálogo", "tama ño":"tamaño", "tambi én":"también", 
                    "t écnica":"técnica", "redacci ón":"redacción", "dispersi ón":"dispersión", "data set":"dataset",
                    "aritm ética":"aritmética", "a través":"através", "construcci ón":"construcción", ", ":", ", 
                    "aplicaci ón":"aplicación", " / ":" y ", "(ecuación 1)":"", "s ímbolo":"símbolo", "gr áfico":"gráfico",
                    "\u200b":"", "\uf0b7":"", " .":".", "..":".", "biol ógica":"biológica", "di álogo":"diálogo",
                    "o tro":"otro", "f ácil":"fácil", "patr ón":"patrón", "as í":"así", " m ":" m", "- ":"-",
                    "i ón":"ión", "unamedida":"una medida", "ingl és":"inglés", "\xa0":" ", "n i":"ni", "n ij":"nij",
                    "ndice":"índice", "...":""}     

    for key in dict4correct.keys():
            text = text.replace(key,dict4correct[key]) 


    if text is not None:
        descrip = text.strip()
        descrip_splitted = descrip.split(".")
        if len(descrip_splitted)>0:
            if ("(" in descrip_splitted[0]) & (len(descrip_splitted)>1):
                if ")" in descrip_splitted[0]:
                    primera_oracion = descrip_splitted[0] + "."
                elif ")" in descrip_splitted[1]:
                    primera_oracion = descrip_splitted[0] + descrip_splitted[1] + "."
                elif len(descrip_splitted)>=3:
                    if ")" in descrip_splitted[2]:
                        primera_oracion = descrip_splitted[0] + descrip_splitted[1] + descrip_splitted[2] + "."
                    else:
                        primera_oracion = descrip_splitted[0] + "."
                else:
                    primera_oracion = descrip_splitted[0] + "."
            else:
                primera_oracion = descrip_splitted[0] + "."
        elif ((descrip[-1] == ",") | (descrip[-1] == ";") | (descrip[-1] == ":")):
            primera_oracion = (descrip[:-1]+'.')
        else:
            primera_oracion = descrip

    new_txt = primera_oracion.capitalize()
    new_txt = clean_sentence_endings(new_txt)
      
    return new_txt


def select1stsentence(df):


    '''
    Function to iterate through a Pandas DataFrame composed of concepts and definitions,
    extracting the first sentence of the latter.
        
    inputs:

    df -> Pandas dataframe composed of concepts and definitions

    outputs:

    df -> Processed pandas dataframe

    '''
    

    idxs2rem = []
    for index, row in df.iterrows():

        if row["descripcion"] is not None:
            descrip = row["descripcion"].strip()
            descrip_splitted = descrip.split(".")
        
        
            if "DisplayStyle".lower() in descrip.lower():
                idxs2rem.append(index)          
            else:

                primera_oracion = Text1stsentence(row["descripcion"])
        else:
            idxs2rem.append(index)  

        df.loc[index, "descripcion"] = primera_oracion
                                
    df = df.drop(idxs2rem)
    df.reset_index(inplace = True, drop = True)

    return df



def RemBadConcepts(df, filter_words=None, first=False):

    '''
    Function to remove pandas dataframe rows with concepts descriptions not fullfiling
    the expected descriptions characteristics
        
    inputs:

    df -> Pandas dataframe composed of concepts and definitions
    filter_words -> Variable containing forbidden words within the concepts definitions

    outputs:

    df -> Processed pandas dataframe
    concepts -> Concepts removed
    indexs -> Indexs removed

    '''    


    if first:
       
        dict4correct = {"gu stos":"gustos", "l ógica":"lógica", "di álogo":"diálogo",
                        "t écnica":"técnica", "redacci ón":"redacción", "dispersi ón":"dispersión", "data set":"dataset",
                        "aritm ética":"aritmética", "a través":"através", "construcci ón":"construcción", ", ":", ", 
                        "aplicaci ón":"aplicación", " / ":" y ", "(ecuación 1)":"",
                        "\u200b":"", "\uf0b7":"", " .":".", "..":"."}
        
        df.loc[:, "descripcion"]  =  df.descripcion.str.replace("“", "").str.replace("”", "").str.replace("  ", " ").str.capitalize()
        
        for key in dict4correct.keys():
            df.loc[:, "descripcion"] = df.descripcion.str.replace(key,dict4correct[key])


    if filter_words == None:
        filter_words = ["publico", "terapia", "ciudad", "Estados Unidos", "universidad", "literatura",
                        "austria", "urban", "cientifico", "presidencia", "bailarina", "artista",
                        "escritora", "España", "contrato", " = ", "Twitter", "Trump", "Hittler"]

    concepts = []
    indexs = []
    
    for index, row in df.iterrows():
        for word in filter_words:
            if word in row["descripcion"]:
                concepts.append(row["concepto"])
                indexs.append(index)

    df = df.drop(indexs)
    df.reset_index(inplace = True, drop = True)

    return df, concepts, indexs



def Manual_WikiSearch(concepts, synonyms, language = "es", languageSite = "es"):


    '''
    Function for manually search on wikipedia, using webscraping techniques,
    for the definitions of specific  data processig related concepts
        
    inputs:

    concepts -> List of concepts for their definitions to be searched
    language -> Language for the concept to be searched
    languageSitet -> wikipedia website language

    outputs:

    results -> Pandas daatframe composed of the input concepts and their found definitions
    notfound -> List composed by the not found concepts definitions

    '''    

    conceptos = []
    descripciones = []
    translated_concepts = []
    notfound = []
    noencontrado = []
    
    for consulta in tqdm(concepts, mininterval=50, total=len(concepts), desc="Searching_Wikipedia", unit="row"):  

        consulta_original = consulta
        consulta = consulta.replace(' ', '_') 
        
        url = f"https://{languageSite}.wikipedia.org/wiki/{consulta}"
        
        definicion = wikisearch(url)

        pattern = r"\[\d+\]"
    
        if (definicion is not None):
    
            if type(definicion) == list:
                definicion = definicion[0]        
            definicion = re.sub(pattern, "", definicion)

            if definicion[0] == ".":
                definicion = definicion[1:]
            definicion = Text1stsentence(definicion)

            if detect(definicion) == "en":
                definicion = translation_function(definicion, english=False)
    
            if language == "es":
                if consulta_original in {v: k for k, v in synonyms.items()}.keys():
                    translated_concepts.append({v: k for k, v in synonyms.items()}[consulta_original])
                else:
                    translated_concepts.append(translation_function(consulta_original, english=True))
                    print(2, ' -- ',  consulta_original, translated_concepts[-1])
                    synonyms[translated_concepts[-1]] = consulta_original
            else:
                if consulta_original in synonyms.keys():
                    translated_concepts.append(synonyms[consulta_original])
                else:
                    translated_concepts.append(translation_function(consulta_original, english=False))
                    print(2, ' -- ',  consulta_original, translated_concepts[-1])
                    synonyms[consulta_original] = translated_concepts[-1]
                                
            conceptos.append(consulta_original)
            descripciones.append(definicion)
        
        else:
            if language == "es":
                noencontrado.append(consulta_original)
                if consulta_original in {v: k for k, v in synonyms.items()}.keys():
                    notfound.append({v: k for k, v in synonyms.items()}[consulta_original])
                else:
                    notfound.append(translation_function(consulta_original, english=True))
            else:
                notfound.append(consulta_original)
                if consulta_original in synonyms.keys():
                    noencontrado.append(synonyms[consulta_original])
                else:
                    noencontrado.append(translation_function(consulta_original, english=False))


    if language == "es":
        results = pd.DataFrame({"concepto": conceptos, "descripcion": descripciones, "concept":translated_concepts, "source":"wikipedia.org"})
    else:
        results = pd.DataFrame({"concepto": translated_concepts, "descripcion": descripciones, "concept":conceptos, "source":"wikipedia.org"})
    
    return results, notfound, noencontrado, synonyms



def SerpAPI_WikiSearch(consultas, keyword4search, API_KEY, chekpoints=100):


    '''
    Function that used the SerpAPI service to search on wikipedia, using google with webscraping techniques,
    for the finding of definitions of specific data processig related concepts
        
    inputs:

    consultas -> List of concepts for their definitions to be searched
    keyword4search -> Complementary keywords for the google wikipedia search
    chekpoints -> Number of implemented seaarches to display the progress (verbose)

    outputs:

    new_df -> Pandas daatframe composed of the input concepts and their found definitions
    notfound -> List composed by the not found concepts definitions

    '''     
    
    notfound = []
    conceptos = []
    definiciones = []
    
    i = 0
    for concepto in tqdm(consultas, mininterval=50, total=len(consultas), desc="Searching_SerpAPI", unit="row"):

        if (i > 1):
            if (i % chekpoints) == 0:
                print(f"\n Checkpoint {i} ******************* \n")
    
        original = concepto
        concepto = re.sub(r"[^\w\s]", "", concepto)
        
        if concepto == "ar":
            new_concept = concepto + " proceso"
        elif concepto == "chip de acelerador":
            new_concept = "Acelerador de IA"
        elif concepto == "adagrad":
            new_concept = concepto
        elif concepto == "condición alineada con el eje":
            new_concept = "Colinealidad"
        elif concepto == "auc":
            new_concept = "auc classifier" 
        elif ((concepto == "magnificación de datos") | (concepto == "aumento")):
            new_concept = "data augmentation"
        elif concepto == "dqn":
            new_concept = "aprendizaje_de_refuerzo_profundo"
        elif concepto == "atributo":
            new_concept = "feature machine learning"
        elif concepto == "agente":
            new_concept = "agente inteligente"    
        elif concepto == "respuesta dorada":
            new_concept = "Golden_record_(informatics)"
        elif concepto == "política codiciosa":
            new_concept = "greedy policy"    
        elif concepto == "red de deep q":
            new_concept = "Aprendizaje_de_refuerzo_profundo"
        elif concepto == "cadena de pensamientos":
            new_concept = "Ingeniería de instrucciones"
        elif concepto == "conjunto de datos con desequilibrio de clases":
            new_concept = "sesgo modelo"
        elif concepto == "reducción de muestreo":
            new_concept = "Submuestreo"
        elif concepto == "red neuronal bayesiana":
            new_concept = "Red bayesiana"
        elif concepto == "bagging":
            new_concept = "Agregación de Bootstrap"
        elif concepto == "condición binaria":
            new_concept = "Lógica binaria"
        elif concepto == "cuadro de límite":
            new_concept = "Límite (matemática)"
        else:
            if "auto" in concepto:
                new_concept = concepto + " datos analisis learning"
            elif "neuro" in concepto:
                new_concept = concepto + " neural network"
            else:
                new_concept = concepto + keyword4search
    
        wikipedia_link = get_wikipedia_link(new_concept, API_KEY)
        
        if not wikipedia_link:
            new_concept = translation_function(concepto, english=True) + " model"
            wikipedia_link = get_wikipedia_link(new_concept, API_KEY)      
        else:
            if ("archivo" in wikipedia_link.lower()) | ("pt" in wikipedia_link.lower()) | (wikipedia_link.split("/")[-1] == 'Machine_learning') | ("Contenido_por_wikiproyecto" in wikipedia_link):
                new_concept = translation_function(concepto, english=True) + " data machine_learning"        
                wikipedia_link = get_wikipedia_link(new_concept, API_KEY)               
            
        if wikipedia_link:
            definicion = wikisearch(wikipedia_link)
            if definicion:
                if detect(definicion) == "en":
                    definicion = translation_function(definicion, english=False)
                pattern = r"\[\d+\]"
                definicion = re.sub(pattern, "", definicion)

                # Imprimir el resultado
                if definicion:
                    conceptos.append(original)
                    definiciones.append(definicion)
            else:
                new_concept = translation_function(concepto, english=False) + " data"
                wikipedia_link = get_wikipedia_link(new_concept, API_KEY)        
                
                definicion = wikisearch(wikipedia_link)
                if definicion:
                    if detect(definicion) == "en":
                        definicion = translation_function(definicion, english=False)
                    pattern = r"\[\d+\]"
                    definicion = re.sub(pattern, "", definicion)

                    # Imprimir el resultado
                    if definicion:
                        conceptos.append(original)
                        definiciones.append(definicion)
                    else:
                        notfound.append(original)
        else:
            notfound.append(original)
        i += 1

    new_df = pd.DataFrame({"concepto":conceptos, "descripcion":definiciones})

    return new_df, notfound



def transformer_similarities(df_base, df_comp, thre=.2):

    '''
    Function that compares definitions of concepts between two datarames. One
    containing the ground truth while the other one has potential complementary
    definitions. The function keeps similaar complementary definitions and
    removes those that are nor enough simiular based on a threshold value
        
    inputs:

    df_base -> Pandas dataframe containing the ground truth concepts definitionn
    df_comp -> Pandas dataframe containing possible complementary concepts definitionn
    thre -> Threshold  value used to dicard or keep complementary definitions

    outputs:

    df -> Pandas dataframe without containing duplicated concepts

    ''' 

    df_best = df_base.copy(deep=True)
    sims = []
    bad_sims = []
    idxs2keep = []
    idxs2rem = []
    count = 0
    limiteper = 1
    
    for index, row in tqdm(df_comp.iterrows(), mininterval=100, total=df_comp.shape[0], desc="Processing_Similarities", unit="row"):
        concept = row["concepto"]
        sen2 = row["descripcion"]
        
        if concept in df_base.concepto.tolist():
            sen1 = df_base[df_base.concepto == concept].descripcion.tolist()[0]
            similarity = transformers_similarity(sen1, sen2)
            sims.append(similarity)      

            if similarity > thre:           
                idxs2keep.append(index)
            else:
                idxs2rem.append(index)
                bad_sims.append(similarity)

        else:
            idxs2keep.append(index)

    return idxs2keep, idxs2rem, sims, bad_sims   



def transformers_similarity(sen1, sen2):

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cuda")
    # Encode sentences
    embedding1 = model.encode(sen1, convert_to_tensor=True)
    embedding2 = model.encode(sen2, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()

    return similarity


def choosing_best_concept(df_base, df_a, df_b):

    '''
    Function that uses transformers to find similarities between sentences, where one
    of them is considered the ground truth, while the other two represent possible
    complementary definitions. The function selects the sentence most similar to
    the ground truth.
        
    inputs:

    df_base -> Pandas daataaframe cotaining the ground truths
    df_a -> Pandas datrame with complementary definitions
    df_b -> Pandas datrame with complementary definitions

    outputs:

    df_best -> Pandas daatframe composed of the the best complementary definitions

    '''         

    df_best = df_a.copy(deep=True)
    
    for index, row in tqdm(df_a.iterrows(), mininterval=50, total=df_a.shape[0], desc="ChoosingBestConcept", unit="row"):
        concept = row["concepto"]
        sen2 = row["descripcion"]
        
        if (concept in df_base.concepto.tolist()) & (concept in df_b.concepto.tolist()):
            sen1 = df_base[df_base.concepto == concept].descripcion.tolist()[0]
            sen3 = df_b[df_b.concepto == concept].descripcion.tolist()[0]       

            similarity1 = transformers_similarity(sen1, sen2)
            similarity2 = transformers_similarity(sen1, sen3)         

            if similarity1 < similarity2:           
                df_a.loc[index, "descripcion"] = sen3

    return df_best  




def find_best_definition(concept, descriptions):

    '''
    Function that uses transformers to find similarities between words and a sentences,
    where the word represents a concept and the sentence a possible definition.
    Depending on the similarity, the concept will be or not with the definition
        
    inputs:

    concept -> Concept to be assiciated with a definition
    definition -> Definitions to be associted with a concept

    outputs:

    best_index -> Indedx representing the position of a definition on a list of definitions
    descriptions[best_index] -> Best definition to be associated with the input concept

    '''   

    # Load a pre-trained model for embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

    concept_embedding = model.encode(concept, convert_to_tensor=True)
    descriptions_embeddings = model.encode(descriptions, convert_to_tensor=True)
    
    # Compute cosine similarity and ensure tensors are moved to CPU before converting to NumPy
    similarities = util.pytorch_cos_sim(concept_embedding, descriptions_embeddings)[0].cpu().numpy()
    
    best_index = np.argmax(similarities)
    
    return best_index, descriptions[best_index]




def removing_duplicates(df, first=False):

    '''
    Function that removes duplicated concepts, keeping the best definition among all
    repeated concepts associated  definitions
        
    inputs:

    df -> Pandas dataframe containing the duplicated concepts

    outputs:

    df -> Pandas dataframe without containing duplicated concepts

    '''       

    df.loc[:, "concepto"]  =  df.concepto.str.lower()
    df.loc[:, "descripcion"]  =  df.descripcion.str.capitalize()
    conceptos = df.loc[df.index[df.duplicated("concepto")]].concepto.unique()
    idxs2rem = []

    for concept in tqdm(conceptos, total=len(conceptos), mininterval=50,desc="Processing duplicates", unit="concept"):
      
        idxs = df[df.concepto == concept].index.tolist()
        definitions = df.loc[idxs, 'descripcion'].tolist()
    
        def2rem = []
        for definition in definitions:
            if len(definition.split()) <= 3:
                def2rem.append(definition)
            elif "(" in definition:
                if ")" not in definition:
                    def2rem.append(definition)
    
        if len(def2rem) > 0:
            for str2del in def2rem:
                definitions.remove(str2del)
    
        if len(definitions) > 0:
            _, best_description = find_best_definition(concept, definitions)
        else:
            best_description = definitions

        try:
            df.loc[idxs[0], "descripcion"] = best_description
            idxs2rem += idxs[1:]
        except:
            idxs2rem += idxs

    df = df.drop(idxs2rem)
    df.reset_index(inplace=True, drop=True)

    return df




def remove_parenthesis_complements(text):
    """
    Removes text within parentheses along with the parentheses themselves.
    """
    cleaned_text = re.sub(r"\([^)]*\)", "", text)
    return cleaned_text.strip()
    
