import requests, os
import pandas as pd
from bs4 import BeautifulSoup, NavigableString, Tag
from http.client import responses
from pypdf import PdfReader
import re, unidecode
from tqdm import tqdm
from Notebooks.Data_Generation.Wikipedia_DataWrangling import Text1stsentence, translation_function
     
def Webscraping1(url):


    headers_types = ['h2', 'h3']
    clases = ['hide-from-toc', None]
    text_ids = ['data-text', 'id']

    response = requests.get(url)

    if (response.status_code < 200) and (response.status_code >= 300):
        print('Error: '+responses[response])

    # Gathering data
    data = response.text

    # Parse the HTML content
    soup = BeautifulSoup(data, 'html.parser')

    # Find all H2 headers
    headers = soup.find_all('p')

    # Defining the lists to contain the gathered data
    concepts = []
    meanings = []

    header1 = None
    header2 = None
    text = []

    for i in range(len(headers) - 1):
        #concepts.append(headers[i])
        if (headers[i].find("h2", class_=clases[0]) is not None) & ((header1 is None)):
            header1 = headers[i].find("h2", class_=clases[0])[text_ids[0]].strip()
            text = []
            concepts.append(headers[i].find("h2", class_=clases[0])[text_ids[0]].strip())

        elif (header1 is not None) & (headers[i].find("h2", class_=clases[0]) is not None):
            text.append(headers[i].text.strip().replace("\n", " "))
            header1 = headers[i].find("h2", class_=clases[0])[text_ids[0]].strip()
            meanings.append(Text1stsentence(''.join(text).replace('0.0', '0').replace('1.0', '1').replace('1.', "")))
            concepts.append(headers[i].find("h2", class_=clases[0])[text_ids[0]].strip())
            text = []        
            if concepts[-1] == "Z-score normalization":
                header1 = None
                text = []
        else:
            text.append(headers[i].text.strip().replace("\n", " "))
            ilast=i

    text.append(headers[i].text.strip().replace("\n", " "))
    meanings.append(Text1stsentence(''.join(text).replace('0.0', '0').replace('1.0', '1').replace('1.', "")))

    df = pd.DataFrame({'concepto':concepts, "descripcion":meanings})
    idxs = df[df.descripcion.str.contains("#")].index
    df = df.drop(idxs)
    df.reset_index(drop=True, inplace=True)
    df.loc[df[df.concepto=="ground truth"].index, "descripcion"] = "Reality, the thing that actually happened."
    df.loc[:, 'concepto'] = df.concepto.str.lower().str.replace('-', ' ').str.replace('_', ' ')

 
    descripciones = []
    conceptos = []
    concepts = df.concepto.str.lower().tolist()
    for index, row in tqdm(df.iterrows(), mininterval=50, total=df.shape[0], desc="Translating2Spanish", unit="row"):
        try:
            translated_concept = translation_function(row["concepto"], english=False)
            translated_desc = translation_function(row["descripcion"], english=False)
            conceptos.append(translated_concept)
            descripciones.append(translated_desc)
        except:
            print("Error: ", row["concepto"], '----',row["descripcion"])
            conceptos.append(row["concepto"])
            descripciones.append('Translation Error')

    df.loc[:, 'concepto'] = conceptos
    df.loc[:, 'descripcion'] = descripciones
    df.loc[:, 'concept'] = concepts
    df.loc[:, 'source'] = [url.split("/")[2]]*len(concepts)
    df.loc[:, 'concepto'] = df.concepto.str.lower()

    synonyms = {}
    for concepto, concept in zip(conceptos, concepts):
        synonyms[concept.lower()] = concepto.lower()

    synonyms['adagrad'] = 'adagrad'
    synonyms['ar'] = "proceso ar"
    synonyms['area under the pr curve'] = 'Área bajo la curva de recuperación de precisión'.lower()
    synonyms['auto regressive model'] = 'modelo autorregresivo'
    synonyms['bag of words'] = 'bolsa de palabras'
    synonyms['language model'] = 'modelo de lenguaje'
    synonyms['overfitting'] = 'sobreajuste'
    synonyms['cloud computing'] = 'computacion en la nube'
    synonyms['plm'] = 'ciclo de vida del producto'
    synonyms['pr auc (area under the pr curve)'] = 'Área bajo la curva de recuperación de precisión'.lower()
    synonyms['unsupervised learning'] = 'aprendizaje no supervisado'
    synonyms['brain computer interfaces'] = 'interfaz cerebro computadora'
    synonyms['arrays'] = 'matrices'
    synonyms['tensorflow'] = 'tensorflow'
    synonyms['optimization'] = 'optimizacion'
    synonyms['mode'] = 'moda'
    synonyms['waterfall model'] = 'modelo en cascada'
    synonyms['directed graph and undidrected graph'] = 'grafos no dirigidos y grafos dirigidos'
    synonyms['google cloud automl'] = 'google cloud automl'
    synonyms['sentiment analysis'] = 'análisis de sentimientos'
    synonyms['kurtosis'] = 'curtosis'
    synonyms['kurtosis coefficient'] = 'coeficiente de curtosis'
    synonyms['boxplot'] = 'diagrama de caja'
    synonyms['confidence interval'] = 'intervalo de confianza'
    synonyms['confidence coefficient'] = 'coeficiente de confianza'
    synonyms['mapreduce'] = 'mapreduce'
    synonyms['pareto chart'] = 'diagrama de pareto'
    synonyms['time series components'] = 'componentes de una serie temporal'
    synonyms['data'] = 'dato'
    synonyms['stratum'] = 'estrato'    
    synonyms['stratified sampling'] = 'muestreo estratificado'    
    synonyms['acid'] = 'acid'    
    synonyms['sharding'] = 'shard'
    synonyms['primary key'] = 'llave primaria'
    synonyms['foreign key'] = 'llave foránea'
    synonyms['denormalization'] = 'denormalización'
    synonyms['rollback'] = 'reversión'
    synonyms['commit'] = 'consolidar'
    synonyms['elasticsearch'] = 'elasticsearch'
    synonyms['cassandra'] = 'cassandra database'
    synonyms['bagging'] = 'bagging'
    synonyms['data governance'] = 'gobernanza de datos'
    synonyms['data government'] = 'gobierno del dato'
    synonyms['quasi variance'] = 'cuasivarianza'
    synonyms['recommendation system'] = 'sistemas de recomendación'
    synonyms['query'] = 'consulta'
    synonyms['queries'] = 'consultas'
    synonyms['training stage'] = 'etapa de entrenamiento'
    synonyms['testing stage'] = 'ettapa de prueba'
    synonyms['validation data set'] = 'conjunto de datos de validación'
    synonyms['training, validation, and  test data sets'] = 'conjuntos de datos de entrenamiento, validación y prueba'
    synonyms['empirical mode decomposition (emd)'] = 'descomposición modal empírica (emd)'
    synonyms['estimator'] = 'estimador'
    synonyms['pipelining'] = 'segmentación'
    synonyms['pipeline'] = 'segmentación'
    synonyms['multiresolution analysis'] = 'análisis multiresolución'
    synonyms['análisis señales multirresolución'] = 'multiresolution analysis'
    synonyms['depthwise separable convolution'] = 'convolución separable en profundidad'
    synonyms['multicloud deployment'] = 'despliegue multinube'
    synonyms['logit'] = 'logit'
    synonyms['odds ratio'] = 'odds ratio'
    synonyms['log odds'] = 'log odds'
    synonyms['charts annotations'] = 'anotaciones en gráficos'
    synonyms['dynamic charts'] = 'gráficos dinámicos'
    synonyms['rosette charts'] = 'gráficos de rosetas'
    synonyms['variation charts'] = 'gráficos de variación'
    synonyms['hexagon chart'] = 'diagrama de hexágonos'
    synonyms['network diagram'] = 'gráficos de red'
    synonyms['gantt charts'] = 'diagramas de gantt'
    synonyms['density plots'] = 'density graphics'
    synonyms['violin plot'] = 'gráficos de violin'
    synonyms['time  series plot'] = 'gráficos de series temporales'    
    synonyms['choropleth maps'] = 'mapas coropléticos'
    synonyms['pie chart'] = 'gráficos de rosca'
    synonyms['radar chart'] = 'gráficos de radar'
    synonyms['bubbles scatter plots'] = 'diagramas de dispersión con burbujas'
    synonyms['area charts'] = 'gráficos de área'
    synonyms['boxplots'] = 'diagramas de caja'
    synonyms['scatter plots'] = 'gráficos de dispersión'
    synonyms['line charts'] = 'gráficos de líneas'
    synonyms['bar charts'] = 'gráficos de barras'
    synonyms['hashing'] = 'hashing'
    synonyms['hinge loss'] = 'pérdida de hinge'
    synonyms['inter-rater reliability'] = 'acuerdo de inter rater'
    synonyms['matrix'] = 'matriz de artículos'  
    synonyms['k means'] = 'k means'
    synonyms['k mean'] = 'k mean'
    synonyms['sparse vector'] = 'vector disperso'     





    for index, row in df.iterrows():
        if row['concept'] in synonyms.keys():
            df.loc[index, 'concepto'] = synonyms[row['concept']]

            
    return df, synonyms


def WebScraping2(url, synonyms):

    '''
    Functions that obtain information from specific webpages

    inputs:

    url -> url from which the data will be retrieved
    header_type -> Header that contains the information to be retrieved
    clase -> class of the header to be retrieved
    text_id ->  Header content containing the title of the information to be retrieved
    paragraphs -> Number of information paragraphs to be retrieved

    outputs:

    df -> Pandas dataframe containing the retrieved data

    '''

    # Establishing connection with the defined URL
    headers_types = ['h2', 'h3']
    clases = ['hide-from-toc', None]
    text_ids = ['data-text', 'id']

    header_type = headers_types[0]
    clase = clases[0]
    text_id = text_ids[1]

    # Establishing connection with the defined URL
    response = requests.get(url)
    if (response.status_code < 200) and (response.status_code >= 300):
        print('Error: '+responses[response])

    # Gathering data
    data = response.text
        
    # Parse the HTML content
    soup = BeautifulSoup(data, 'html.parser')
     
    # Extract information related to the defined header
    headers = soup.find_all('h3')

    # Defining the lists to contain the gathered data
    concepts = []
    meanings = []

    # Retrieving Data
    for header in headers:
        concepts.append(header[text_id])
        nextNode = header
        description  = []
        while True:
            nextNode = nextNode.nextSibling
            if nextNode is None:
                break
            if isinstance(nextNode, NavigableString):
                text = nextNode.strip()
                if '#' not in text:
                    description.append(text)            
            if isinstance(nextNode, Tag):
                if nextNode.name == header_type:
                    break
                text = nextNode.get_text(strip=False).strip()
                if '#' not in text:
                    description.append(text)

        meanings.append(Text1stsentence(''.join(description).replace('\n', ' ')))

    # Generating a pandas dataframe composed of the gathered data

    df = pd.DataFrame({'concepto':concepts, 'descripcion':meanings})
    df.loc[:, 'concepto'] = df.concepto.str.replace("-", " ").str.replace("_", " ")
    df.loc[df[df.concepto=='r'].index, 'concepto']  = "lenguaje r"



    df.loc[:, 'concept'] = df['concepto'].map({v: k for k, v in synonyms.items()})
    df.loc[:, 'source'] = [url.split("/")[2]]*len(concepts)
    print()
    for idx in tqdm(df.loc[df.index[df.concept.isna()]].index, mininterval=50, total=df.loc[df.index[df.concept.isna()]].index.shape[0], desc="Translating2English", unit="row"):
    #for idx in df.loc[df.index[df.concept.isna()]].index:
        try:
            translated_concept = translation_function(df.loc[idx, 'concepto'], english=True)
            df.loc[idx, 'concept'] = translated_concept
        except:
            print('Error: ',  df.loc[idx, 'concepto'])
    df.loc[:, 'concept'] = df.concept.str.lower()
    

    for concept, concepto in zip(df.concept, df.concepto):
        synonyms[concept.lower()] = concepto.lower()
   

    return df, synonyms




def GeneratingData_WebScraping(URLs, Headers, Clases, Text_Ids, Paragraphs):

    '''
    Function that retrieve data from different URLs relying on the webscraping function

    inputs:

    URLs -> Lists containing urls from where the data will retrieved
    Headers -> Headers from where the data will be retrieved, related to their corresponding urls
    Clases -> Classes of the headers to be retrieved
    Text_Ids ->  Titles for the information to be retrieved from the selected headers
    Paragraphs -> Number of paragraphs to be retrieved for each header

    outputs:

    df -> Dataframe composed by the data retrieved from all the defined URLs

    '''

    for n in range(len(URLs)):
        if n ==0:
            #df = WebScraping(URLs[n], Headers[n], Clases[n], Text_Ids[n])
            df, synonyms = Webscraping1(URLs[n])
        else:
            df_tmp, synonyms = WebScraping2(URLs[n], synonyms)
            df = pd.concat([df, df_tmp], ignore_index=True, sort=False)

    return df, synonyms



def GatheringData_PDFReader1(path):

    '''
    Function to retrieve information for a pdf

    inpurt:

    path -> Path containing the location for the pdf file from which the  data will retrieved

    outputs:

    df -> Pandas dataframe compose of the retrieved data

    '''
    

    # Generating the pdf reader
    reader = PdfReader(path)

    # Initializing the lists will be composed of the gathered data
    concepts = []
    meanings = []

    # Extracting data from the selected pages
    for i in range(1,len(reader.pages)):

        # getting a specific page from the pdf file
        page = reader.pages[i]

        # extracting text from page
        text = page.extract_text()

        # Retrieving information
        for concept in text.split('●')[1:]:

            if len(concept.split('.')) > 1:
                if '(' in concept.split('. ')[0]:
                    concepto = concept.split('). ')
                else:
                    concepto = concept.split('. ')
                if '(' in concepto[0]:
                    concepts.append(concepto[0].split('(')[0].lstrip())                
                elif '/' in concepto[0]:
                    words = concepto[0].split('/')
                    if len(words[0]) < len(words[1]):
                        concepts.append(words[0].lstrip())
                    else:
                        concepts.append(words[1].lstrip())
                else:
                    concepts.append(concepto[0].lstrip())

                meanings.append(concepto[1].replace('\n', '').replace('\t', '').lstrip())

    # Generating a pandas dataframe composed of the gathered data}
    source = ["book1"]*len(concepts)    
    df = pd.DataFrame({'concepts':concepts, 'definitions':meanings, "source":source})

    return df

        

def GatheringData_PDFReader2(path):

    '''
    Function to retrieve information for a pdf

    inpurt:

    path -> Path containing the location for the pdf file from which the  data will retrieved

    outputs:

    df -> Pandas dataframe compose of the retrieved data

    '''

    reader = PdfReader(path)
    
    unused = ['definición', 'historia', 'primeros', 'primeras', 'empezando', 'aplicaciones']
    forbiddend_des = ['capítulo', 'anterior']
    
    conceptos = []
    definiciones = []
    
    for page in range(38,250):

        try:
            page = reader.pages[page]
            text = page.extract_text()
            
            indexs = []
            for i, txt in enumerate(text.split('\n')):

                if (len(txt)>0) & ((i+1)<len(text.split('\n'))):
                    if len(text.split('\n')[i+1])>0:
                        if (txt[0].isdigit()) & (txt[1]=='.') & (~text.split('\n')[i+1][0].isdigit()):
                            if (txt.split(' ')[1][0].lstrip().isalpha()):
                                indexs.append(i)
                              
            
            for n in range(0,len(indexs)): 
        
                concepto = ' '.join(text.split('\n')[indexs[n]].split(' ')[1:])

                if indexs[n]+1<=indexs[-1]:                                  
                    definicion = ' '.join(text.split('\n')[(indexs[n]+1):indexs[n+1]]).replace('.', '..%.').split('.%.')[:2][0]
                else:
                    definicion = ' '.join(text.split('\n')[(indexs[n]+1):]).replace('.', '..%.').split('.%.')[0:2][0]
        
                Valid_Flag = True
                for word in unused:
                    if word in unidecode.unidecode(concepto.lower()):
                        Valid_Flag = False
    
                for word in forbiddend_des:
                    if word in unidecode.unidecode(definicion.lower()):
                        Valid_Flag = False                    
        
                if Valid_Flag:
                    conceptos.append(concepto)
                    definiciones.append(definicion)
    
        except:
            test_a = 1
    
    source = ["book2"]*len(conceptos)
    df = pd.DataFrame({'concepts':conceptos, 'definitions':definiciones, "source":source})
    df.drop_duplicates(subset='concepts', keep='last', inplace=True)

    return df



def GatheringData_PDFReader3(path):

    '''
    Function to retrieve information for a pdf

    inpurt:

    path -> Path containing the location for the pdf file from which the  data will retrieved

    outputs:

    df -> Pandas dataframe compose of the retrieved data

    '''

    reader = PdfReader(path)
   
    unused = ['definición', 'historia', 'primeros', 'primeras', 'empezando', 'aplicaciones']
    forbiddend_des = ['capítulo', 'anterior']
    
    conceptos = []
    definiciones = []
    for page in range(13,75):

        try:
            
            page = reader.pages[page]
            text = page.extract_text()
                        
            if '4.1.1' in text:
                text = text.replace('4.1.1', '4.1.1 ')
            
            indexs = []
            for i, txt in enumerate(text.split('\n')):
                if (len(txt)>0):
                    if (txt[0].isdigit()) | (txt[0]=='•'):
                        if len(indexs)>0:
                            if (i-1) != indexs[-1]:
                                indexs.append(i)
                        else:
                            indexs.append(i)                    
    
            for n in range(0,len(indexs)): 
    
                concepto = ' '.join(text.split('\n')[indexs[n]].split(' ')[1:])
    
                if len(concepto)>0:       
        
                    if indexs[n]+1<=indexs[-1]:   
                        definicion = ' '.join(' '.join(text.split('\n')[(indexs[n]+1):indexs[n+1]]).replace('.', '..%.').split('.%.')[:2][0:3])
                    else:
                        definicion = ' '.join(' '.join(text.split('\n')[(indexs[n]+1):]).replace('.', '..%.').split('.%.')[0:2][0:3])
    
                    if '(' in concepto:
                        concepto = concepto.split('(')[0]
                    
                    if (':' in concepto):
                        remanent = concepto.split(':')[1:]
                        concepto = concepto.split(':')[0]
                        definicion = remanent[0] + ' ' + definicion
    
                    if ('.' in concepto):
                        remanent = concepto.split('.')[1:]
                        concepto = concepto.split('.')[0]
                        definicion = remanent[0] + ' ' + definicion
                          
                    if 'figura' in definicion.lower():
                        definicion = definicion.lower().split('figura')[0]
                        remanent = concepto.split(':')[1:]
                        concepto = concepto.split(':')[0]
                        definicion = remanent[0] + ' ' + definicion                    
                        
                    
                    Valid_Flag = True
    
                    if ('[' in concepto):
                        if concepto.replace('[', '').replace(']', '').isdigit():
                            Valid_Flag = False
                    
                    for word in unused:
                        if word in unidecode.unidecode(concepto.lower()):
                            Valid_Flag = False
        
                    for word in forbiddend_des:
                        if ('❖' in definicion) | ('•' in definicion):
                            Valid_Flag = False 
                        if word in unidecode.unidecode(definicion.lower()):
                            Valid_Flag = False                    
            
                    if (len(concepto)==0) | (len(definicion)==0):
                        Valid_Flag = False
    
                    if len(concepto) > len(definicion):
                        Valid_Flag = False
    
                    if Valid_Flag:
                        conceptos.append(concepto)
                        definiciones.append(definicion)                       

        except:
            test_a = 1


    source = ["book3"]*len(conceptos)
    df = pd.DataFrame({'concepts':conceptos, 'definitions':definiciones, "source":source})
    df.drop_duplicates(subset='concepts', keep='last', inplace=True)

    return df




def GatheringData_PDFReader4(path):

    '''
    Function to retrieve information for a pdf

    inpurt:

    path -> Path containing the location for the pdf file from which the  data will retrieved

    outputs:

    df -> Pandas dataframe compose of the retrieved data

    '''



    reader = PdfReader(path)       

    unused = ['definición', 'historia', 'primeros', 'primeras', 'empezando', 'aplicaciones']
    forbiddend_des = ['capítulo', 'anterior']
    concepts = []
    descriptions = []

    for i in range(4,66):

        page = reader.pages[i]
        text = page.extract_text()
        #text = unidecode.unidecode(text)
        conceptos = []
        definiciones = []

        test = [re.findall(r'\(.*?\)',txt[-8:]) for txt in text.split('.-')]
        for item in test:
            if len(item)>0:
                text = text.replace(item[0], '')

        test = [txt[-8:] for txt in text.split('.-')] 

        try:
            for txt in text.split('.-'):
                words = []
                for word in re.findall('([A-Z]+)', txt):
                    if len(word)>1:
                        words.append(word)

                
                # Hardcoded conditions - Need time to automate this finding
                concepto = []

                if 'DISTRIBUCIÓN NORMAL CURVA NORMAL' in ' '.join(words):
                    words = ['DISTRIBUCIÓN NORMAL O CURVA NORMAL']   
                    concepto = 'DISTRIBUCIÓN NORMAL O CURVA NORMAL'  
                    

                if 'COEFICIENTE DE ASIMETRÍA DE FISHER' in ' '.join(words):
                    words = ['COEFICIENTE DE ASIMETRÍA DE FISHER']   
                    concepto = 'COEFICIENTE DE ASIMETRÍA DE FISHER'     

                if 'COEFICIENTE DE ASIMETRÍA DE PEARSON' in ' '.join(words):
                    words = ['COEFICIENTE DE ASIMETRÍA DE PEARSON']   
                    concepto = 'COEFICIENTE DE ASIMETRÍA DE PEARSON'                   

                if 'COEFICIENTE DE CONTINGENCIA' in ' '.join(words):
                    concepto = 'COEFICIENTE DE CONTINGENCIA Chi-Cuadrado'
                    words = ['COEFICIENTE DE CONTINGENCIA Chi-Cuadrado']
                                
                if 'ASIMETRÍA' in words:
                    concepto = 'ASIMETRÍA'
                    words = ['ASIMETRÍA']
                
                if len(concepto)<1:
                    concepto = ' '.join(words)
              
                idx = text.find(concepto)
        
                if (text[idx:idx+len(concepto)+2][-2:] == '.-') | (text[idx:idx+len(concepto)+3][-3:] == ' .-'):   
                    conceptos.append(' '.join(words))
                    
            idxs = []
            for concepto in conceptos:
                idxs.append((text.find(concepto), text.find(concepto)+len(concepto)+2)) 
            
            for n in range(0,len(idxs)):
                if n < (len(idxs)-1):
                    descripcion = ' '.join(text[idxs[n][1]:idxs[n+1][0]].replace('.', '..%.').split('.%.')[:2])             
                else:
                    descripcion = text[idxs[len(idxs)-1][1]:].replace('.', '..%.').split('\n')[0].split('.%.')[0]
                  
                if (':' in descripcion) | ('=' in descripcion):  
                    if ':' in descripcion:
                        idx1  = descripcion.index(':')              
                    elif '=' in descripcion:
                        idx1 = descripcion.index('=')
                 
                    if ('.' in descripcion[:idx1-1]) & (',' in descripcion[:idx1-1]):
                        idx3 = descripcion[:idx1-1].rindex('.')  
                        idx2 = descripcion[:idx1-1].rindex(',')
                        if idx2 > idx3:
                            idx2 = idx2
                        else:
                            idx2 = idx3                    
                    elif '.' in descripcion[:idx1-1]:
                        idx2 = descripcion[:idx1-1].rindex('.')
                    elif ',' in descripcion[:idx1-1]:
                        idx2 = descripcion[:idx1-1].rindex(',')
                    else:
                        idx2 = idx1-1        
                    
                    descripcion = descripcion[:idx2]

                definiciones.append(descripcion)

        except:
            test_a = 1

        for nk in range(len(conceptos)):
            if conceptos[nk][0] == conceptos[nk][1]:
                conceptos[nk] = conceptos[nk][1:]
            concepts.append(conceptos[nk])
            descriptions.append(definiciones[nk])
            
        source = ["book4"]*len(concepts)
        df = pd.DataFrame({'concepts':concepts, 'definitions':descriptions, "source":source})
        df.loc[:, 'concepts'] = df.concepts.str.replace("ndice","índice")
        idxs  =  df[df.definitions.str.contains(" \(por ejempl")].index
        df.loc[idxs, 'definitions'] = df.loc[idxs].definitions.str.replace(" (por ejempl", ".").tolist()        
        df.drop_duplicates(subset='concepts', keep='last', inplace=True)

    return df




    
