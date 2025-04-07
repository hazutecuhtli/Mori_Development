import re, unidecode, nltk, spacy
import pandas as pd
import regex as re
import numpy as np
from nltk.corpus import stopwords
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Auto-download if missing
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    
nltk.download('stopwords')
stopwords = set(stopwords.words('spanish'))

# Defining words and punctuations to be removed or to be used as index for filtering text
forbiddenwrds = ['en el que se cumple', 'en lo que se cumple', ' mediante', 'lo que incluye', 'por ejemplo', 'como los siguientes',
                 'una o más de las siguientes', 'una de las siguientes', ', que se basan en', ' se incluyen los siguientes',
                 'con el fin de', 'y en caso de', 'y se pueden', ', es decir', ', por lo tanto', ', y', ', aunque si',
                 ', en donde', ', así también', 'consulta']

wrds2remove = ['en términos generales,', ', por lo general,', 'en el aprendizaje por refuerzo,', 'por lo general', 'según wikipedia,',
               'hadoop distributed file system conocido también por sus siglas hdfs el cual ', '<<', '>>',
               'en estadística grados de libertad de un estadístico calculado en base a n datos,', 'sinónimo de ',
               'termino sobrecargado con alguna de las siguientes definiciones:', 'sinónimo de función.']

separators = [':', ';', ',', ' y ']

wrds2fix = [' s ', ' n ', ' on', ' tico ']

paremcases = ["figura", "Figura", "ecuación", "Ecuación", "\\(", "\\)", "et al", "según"]



def clean_sentence_endings(text):
    
    # Regex pattern to match "(letter." at the end of a sentence
    pattern = r'\s*\([A-Za-z]\.\s*$'  # Matches " (a." or " (B." etc.
    
    # Remove the matching part from the text
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text



def tokenizer(texto):

    '''
    Function that tokenize sentences, based on the regex module

    input:

    texto -> String to tokenize

    output:

    output -> Tokens found

    '''
    
    return re.findall(r'[\w-]*\p{L}[\w-]*', texto)


def removepar(txt):

    '''
    Function that filter out text, based on the presence of ()

    input:

    txt -> Text to process

    output:

    txt -> Processed text

    '''
    
    tmp = re.findall(r'\(.*?\)', txt)
    if len(tmp) > 0:
        for word in paremcases:
            if word in txt:
                return remove_parenthesis_complements(txt)
    else:
        return txt



def removebrackets(txt):

    # Regex pattern to remove expressions like [number]
    pattern = r'\s*\[\d+\]'

    # Apply regex substitution to remove citations
    txt = re.sub(pattern, '', txt)

    return txt
    


def fix_dot_spacing_issues(text):
    # Regular expression to find dots not followed by a space or end of paragraph
    pattern = r"\.(?!\s|$)"
    
    # Replace incorrect cases with '. '
    fixed_text = re.sub(pattern, ". ", text)
    
    return fixed_text



def remove_parenthesis_complements(text):
    """
    Removes text within parentheses along with the parentheses themselves.
    """
    cleaned_text = re.sub(r"\([^)]*\)", "", text)
    return cleaned_text.strip()


def removepar(txt):

    '''
    Function that filter out text, based on the presence of ()

    input:

    txt -> Text to process

    output:

    txt -> Processed text

    '''
    pattern = r'\((.*?)\)'
    paren_content  = re.findall(pattern, txt)
    for content in paren_content:
        if (len(content.split(' ')))>12:
            txt = txt.replace('('+content+')', '')
        for word in paremcases:
            if word in txt:
                return remove_parenthesis_complements(txt)
            else:
                return txt
        
    return txt  
        

def Cleaning_ManualGathered(df, col1, col2):

    '''
    Function to improve text realted to data processing concepts

    input:

    df -> Pandas dataframe containing the text to be improved
    col1 -> Column dataframe name containig the concepts
    col2 -> Column dataframe name containig the definitions

    output:

    df -> Processed dataframe

    '''    

    # Fixing words that affect the cleansing processes
    df.loc[:, col2] = df[col2].str.replace('0.0', '0').str.replace('1.0', '1')
    df.loc[:, col2] = df[col2].str.replace('Sinónimo de función.', '')
    df.loc[:, col2] = df[col2].str.replace('(Ver indice de concentracion de Gini). ', '')
    #df, concepts2review = BasicData_Cleaning(df, col1, col2)
    #df = select1stsentence(df)

    pattern = r'\s*\[\d+\]'
    df.loc[:, col2] = df[col2].apply(lambda x: re.sub(pattern, '', x))

    df.loc[:, col2] = df[col2].apply(lambda x: removepar(x))

    pattern = r'\s*\([^)]*$'
    df.loc[:, col2] = df[col2].apply(lambda x: re.sub(pattern, '', x))
    
    # Determining the number of tokens per definition
    df.loc[:, 'num_tokens'] = df[col2].apply(lambda x: nlp(x)).str.len()

    # Defining limits when getting more than one sentence for a description
    min_ = 30       # Minimun size for the first splitted sentence in a paragraph
    max_ = 200      # Maximum size for the secodnssplitted sentence in a paragraph
    min2 = 50

    # Creating the space to store data for the cleansing processes
    idxs2drop = []    
    new_descriptions = []
    tokens = []
    df.loc[:, 'tokens'] = np.nan

    # Iterating through all the dataframe rows
    for index, row in df.iterrows():

        # Fixing words within the retrieved data sentences
        text = removepar(row[col2])

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

        # Removing sentences that are not definitions but stories related to concepts
        if 'investigación' in text:
            idxs2drop.append(index)

        # Fixing words 
        for wrd in wrds2fix:
            text = text.replace(wrd, wrd.lstrip())

        # Removing words or expressions 
        for wrd in wrds2remove:
            text = text.replace(wrd, '')

        # Filtering out text that contains certain expressions
        for wrd in forbiddenwrds:
            if wrd in text:
                text = text[:text.rfind(wrd)]
       

        # Processing text if, even after the previous cleaning process, it remains with more than 40 tokens
        if row['num_tokens']>40:
            # Splitting sentences based on the presence of specific punctuation
            for separator in separators:
                if separator in text:
                    if separator == ',':
                        if 'y' in text:
                            idx1 = text.find(' y ')
                            idx2 = text.rfind(separator)
                            if idx1 < idx2:
                                text = text[:idx2]
                        else:
                            text = text[:text.rfind(separator)]
                    elif separator == ' y ':
                        idx1 = text.find(' y ')
                        if text.rfind(' y ') != idx1:
                            text = text[:text.rfind(' y ')]  
                    else:
                        text = text[:text.rfind(separator)]                
                        if len(text)<=max_:
                            break            

        # Determining which sentences to remove                
        if index not in idxs2drop:
            df.loc[index, col2] = text
            tokns = [token.text for token in nlp(text)]      

    # Removing sentences
    #df.drop(idxs2drop, axis=0, inplace=True)    

    return df[df.columns[:3]]


def BasicData_Cleaning(df, col1, col2):

    '''
    Function that prepare the data with actions such as converting to lowercases, defining sentences
    among other basic actions

    inputs:

    df -> pandas dataframe containing at least two columns
    col1 -> Column composed by data processing concepts
    col2 -> Columns containing the descriptions for the concepts stored in Col1

    outputs:

    df -> Pandas dataframe containing the cleaned data
    concepts2modify -> List containing concepts to be reviewed

    '''
    
    # Removing accents, apostrophe and similar characters
    #df.loc[:, col1] = df[col1].apply(lambda x: unidecode.unidecode(x))
    #df.loc[:, col2] = df[col2].apply(lambda x: unidecode.unidecode(x))

    # Converting concepts to lowercase and determining sentences within the stored descriptions
    df.loc[:, col1] = df[col1].str.lstrip().str.lower().str.rstrip().str.replace('  ', ' ')
    df.loc[:, 'sentences'] = [len(text.replace('.', '..%.').split('.%.')) for text in df[col2]]

    # Selecting sentences within the stored descriptions with the required characteristics
    for sentence in range(3):
        idxs = df[df.sentences>=(sentence+1)].index
        df.loc[:, 'words_in_sentence_'+str(sentence+1)] = 0
        df.loc[idxs, 'words_in_sentence_'+str(sentence+1)] = [len(words.replace('.', '..%.').split('.%.')[sentence].split(' ')) for words in df.loc[idxs,col2]]
        df['words_in_sentence_'+str(sentence+1)] = df['words_in_sentence_'+str(sentence+1)].astype(int)

    # Defining lists to stored the data to be cleaned, then it will be used to generate
    # pandas dataframe
    concepts2improve = []
    idx2drop = []
    concepts2modify = []

    # Cleaning Data
    for index, row in df.iterrows():

        # Determining concepts that need to be fixed externally
        if '(' in row[col1]:
            concepts2modify.append(index)

        # Selecting descriptions of concepts composed of more than 1 word
        if row['words_in_sentence_1']==1:
            tag1 = True
            for col in df.columns[-2:]:
                if df.loc[index, col]>1:
                    df.loc[index, col2] = row[col2].split('.')[1].lstrip().rstrip()
                    Tag1 = False
                    break
                if Tag1:
                    concepts2improve.append(row[col1])
                    idx2drop.append(index)
            
        # Removing descriptions containing equations
        elif ('\\' in row[col2].split('.')[0]) | ('consulta'==row[col2].lower().split('.')[0].split(' ')[0]):
            concepts2improve.append(row[col1])
            idx2drop.append(index)

        # Defining the first sentence within the descriptions to be used to describe
        # the concepts
        else:
            df.loc[index, col2] = row[col2].replace('.', '..%.').split('.%.')[0].lstrip().rstrip()
            
        # Removing concepts with undefined characters
        if '@' in row[col1]:
            idx2drop.append(index)


    # Removing rows withtin the input dataframe that not follows the required characteristics
    df.drop(idx2drop, axis=0, inplace=True)
    # Removing columns created for the cleansing processes
    df.drop(df.columns[-4:], axis=1, inplace=True)    
    
    # Removing parentheses within the defined concept. and combine their content with the rest of the concept text
    df.loc[:, col2] = df[col2].str.lstrip().str.lower().str.rstrip().str.replace('  ', ' ')
    idxs_par = df[df[col1].str.contains("\\(")].index
    df_tmp = df[df[col1].str.contains("\\(")].copy(deep=True)
    df_tmp.reset_index(drop=True, inplace=True)
    df_tmp.loc[:,col1] = [re.findall(r'\(.*?\)', concept)[0].replace('(', '').replace(')', '') for concept in df_tmp[col1]]    
    df.loc[idxs_par,col1] = [(re.sub("\\(.*?\\)","()",concept).replace('()', '').rstrip()) for concept in df.loc[idxs_par,col1]]
    df = pd.concat([df, df_tmp], ignore_index=True, sort=False)
    df.reset_index(drop=True, inplace=True)
    #df.loc[:, col2] = [' '.join(des) for des in df[col2].map(lambda oracion: tokenizer(oracion))]

    return df, concepts2modify



def Descriptions_Standardization(df, col_des, min_size, max_size):

    '''
    Function that implements a second cleaning stage, and that standardize descriptions, based on the number of
    words contained within them, range defined by the min_size and max_size variables
    
    inputs:

    df -> pandas dataframe containing at least two columns
    col_des -> Columns containing the descriptions
    min_size -> Variable that defines the range minimum value
    max_size -> Variable that defines the range maximun value

    outputs:

    df -> Pandas dataframe containing the standardized data

    '''
    df.loc[:, 'tmp'] = ''
    # Calculating the current number of words trhat compose the descriptions
    df.loc[:,'tamano'] = [len(desc.split(' ')) for desc in df[col_des]]
    idxs = df[df['tamano']>((min_size+max_size)/2)].index

    # Cleaning and standardization of the concepts descriptions
    for index, row in df.iterrows():

        df.loc[index, 'desc'] = row[col_des]

        # Finding sentences composed of more than one idea, and selecting the
        # first one to describe its related concept
        if len(row[col_des].split('.')) > 1:
            df.loc[index, col_des] = df.loc[index, col_des].split('.')[0]

        # Removing composed specific words that do not contribute to the concept description
        if ' '.join(df.loc[index, col_des].split(' ')[0:2]) in ['es una', 'es la', 'es el', 'es un']:
            df.loc[index, col_des] = ' '.join(df.loc[index, col_des].split(' ')[2:])

        # Removing single specific words that do not contribute to the concept description
        if ' '.join(df.loc[index, col_des].split(' ')[0:1]) in ['una', 'la', 'el', 'un']:
            df.loc[index, col_des] = ' '.join(df.loc[index, col_des].split(' ')[1:])

        # Reducing the number of words within descriptions by removing parentheses information
        #if ('(' in df.loc[index, col_des]) and (len(df.loc[index, col_des])>150):
        #    df.loc[index, col_des] = re.sub("\\(.*?\\)","()",df.loc[index, col_des]).replace('()', '').rstrip()

        

    # Filtering out descriptions that not follow the desired number of words within them
    df.loc[:,'tamano'] = [len(desc.split(' ')) for desc in df[col_des]]
    df = df[(df['tamano']>=min_size) & (df['tamano']<=max_size)]#[df.columns[:2]]
    df.reset_index(drop=True, inplace=True)
    df = df.rename(columns = {col_des:'contexto', 'desc':'descripcion', 'Source':'source'})
    

    return df[[df.columns[0], 'descripcion', 'contexto', 'source']]
    

