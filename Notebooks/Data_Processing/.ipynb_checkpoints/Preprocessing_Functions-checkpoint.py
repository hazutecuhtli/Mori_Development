
import pandas as pd


# Grammatical articles to use, on their respective escenarios
dict_articles = {'ion':'la', 'ote':'un', 'red':'una', 'is':'el', 'as':'las', 'es':'las', 'os':'los', 'en':'un',
                 'el':'un', 'ud':'la', 'iz':'la', 'ia':'la', 'or':'el', 'ad':'la','o':'el', 'a':'la', 'e':'el'}

# Reserved words, these words won't be considered for their use with grammatical articles
nonvalid_wrds = ['MongoDB', 'Cassandra', 'Redis', 'DynamoDB', 'Couchbase', 'BigQuery', 'RDS', 'S3', 'MySQL', 
                 'Azure', 'SQL', 'SQLite', 'phpMyAdmin', 'postgresql', 'DBeaver', 'HeidiSQL', 'pgAdmin', 'Robo 3T',
                 'Python', 'R', 'Java', 'C#', 'JavaScript', 'C++', 'Pandas', 'NumPy', 'SQLAlchemy', 'Julia',
                 'PyODBC', 'Dask', 'Hadoop', 'Apache', 'Spark', 'PySpark', 'Excel',  'API', 'Alteryx', 'Tableu', 'PowerBI',
                 'Azure', 'Watson', 'AWS', 'Databrikcs', 'NodeRed', 'saas', 'paas', 'iaas']
nonvalid_wrds = [tool.lower() for tool in nonvalid_wrds]


def Generating_Parafrasis(df, Inquiries_Templates):

    '''
    Function to extract compoenents of a json file composed by questions and answers
    related to the processing of data
    
    inputs:

    df -> Pandas dataframe containing que questions, asnwers, and the gramatical article related to the question
    Inquiries_Templates -> Questions templates for the increments of the questions cases

    outputs:

    intents -> Python list containing questions and anwers related to the processing of data
    
    '''
    # Creating the space to store the questions or intents
    Chatbot_Dataset = {}
    intents = []

    # Generating the intests composed by questions and asnwers related to processing data
    for index, row in df.drop_duplicates("concepto").iterrows():
        for question_template in Inquiries_Templates:
            if question_template[1]=='si':
                if question_template[2]!='0':
                    #print(row["articulo"].strip())
                    if row["articulo"].strip() in ['las', 'los']:
                        intents.append({"canonical_term": row["concepto"],
                                        "input": question_template[2]+row["articulo"]+row["concepto"]+'?',
                                        "output":row["descripcion"],
                                        "context": row["contexto"]})
    
                    else:
                        intents.append({"canonical_term": row["concepto"],
                                        "input": question_template[0]+row["articulo"]+row["concepto"]+'?',
                                        "output":row["descripcion"],
                                        "context": row["contexto"]})
            else:
                intents.append({"canonical_term": row["concepto"],
                                "input": question_template[0]+' '+row["concepto"],
                                "output":row["descripcion"],
                                "context": row["contexto"]})    

    return intents


            

def FromJson2Pandas(data, preguntas, articulos):

    '''
    Function to extract compoenents of a json file composed by questions and answers
    related to the processing of data
    
    inputs:

    data -> file containing the questions and answers
    preguntas -> Expected questions structure to get concepts from questions
    articulos-> Grammar rules to define words that precede nouns

    outputs:

    df -> Pandas dataframe containing the articles related to the col input
    
    '''
    
    # Defining the lists spaces to store the different components found within the json file
    conceptos = []
    articles = []
    conectores = []
    descriptions = []
    etiquetas = []
    context = []

    # Extracting components tags from the json file    
    for intent in data:
        
        flag_arts = False
        flag_quest = False
        patterns = intent.get("input", [])
        context.append(intent.get("context", []))
        descriptions.append(intent.get("output", []))
    
        for pregunta in preguntas:
            if pregunta in patterns:
                flag_quest = True
                patterns = patterns.replace(pregunta, '')
                for article in articulos:
                    if article == patterns.split()[0]: 
                        flag_arts = True
                        articles.append(' '+patterns.split()[0].strip()+' ')
                        conceptos.append(' '.join(patterns.split()[1:]).replace('?', '').strip())
        
        if not flag_arts:
            articles.append(' ')
            conceptos.append(patterns.replace('?', '').strip())
        if not flag_quest:
            a =2 
        
    df = pd.DataFrame({'concepto':conceptos, 'descripcion':descriptions, 'contexto':context, 'articulo':articles})

    return df

def Finding_GrammarArticles(df, col, articles=dict_articles, nonvalid_wrds=nonvalid_wrds):

    '''
    Function that creates an additional column to the input dataframe, containing grammatical rules
    that precede data in the col input
    
    inputs:

    df -> pandas dataframe containing at least two columns
    col_des -> Columns containing the concepts to process
    articles-> Grammar rules to define words that precede nouns
    nonvalid_wrds -> Words that are not related to grammatical articles, since those are restricted

    outputs:

    df -> Pandas dataframe containing the articles related to the col input

    '''
    
    # Creating a column composed of the concepts 1st word
    df.loc[:, 'concept_1stwrd'] = [word.split(' ')[0] for word in df[col]]
    # Space to store the genartion of noun articles
    df.loc[:, 'word_article'] = ['']*df.shape[0]
    
    total = 0
    idxs = []

    # Determining if the first words of the input concepts need to have a grammatical article
    for key in dict_articles.keys():
        
        idx = df[(df.concept_1stwrd.str[-len(key):]==key) & ~(df.concept_1stwrd.isin(nonvalid_wrds) & (df.concept_1stwrd.str.len()>3))].index.tolist()    
        idx = [ix for ix in idx if ix not in idxs]
        
        df.loc[idx, 'word_article'] = dict_articles[key]
        idxs += idx

    return df[df.columns[0:3].tolist()+['word_article']]




def Rasa_Dataset(df, col, Intents_Templates):

    '''
    Function that generate fields within the input dataframe containing intests for the col column,
    having as an objetive the generation of intents to be used within the Rasa framwork
    
    inputs:

    df -> pandas dataframe containing at least two columns
    col_des -> Columns containing the concepts to be used for the intents generation
    Intents_Templates -> Templates for the generation of intents

    outputs:

    df -> Pandas dataframe containing information needed to train a Rasa bot

    '''
    
    # Determining the grammatical articles to use for the input concepts
    df = Finding_GrammarArticles(df, col)

    # Creating the concepts intents
    for i, template in enumerate(Intents_Templates):

        # Reviewing grammatical rules for the use of the intent templates
        intents = []
        for index, row in df.iterrows():

            
            if ('es'==template[0][-2:]) & (row['word_article']!=''):
                if (row['word_article'][-1]!='s'):
                    intents.append(template[0] + ' ' + row['word_article']+ ' ' +row[col]+template[1])
                else:
                    intents.append(template[0][:-2] + 'son' + ' ' + row['word_article']+' '+row[col]+template[1])

            elif ('ve'==template[0][-2:]) & (row['word_article']!=''):
                if (row['word_article'][-1]!='s'):
                    intents.append(template[0] + ' ' + row['word_article']+ ' ' +row[col]+template[1])
                else:
                    intents.append(template[0][:-5] + 'sirven' + ' ' + row['word_article']+' '+row[col]+template[1])

            elif ('ca'==template[0][-2:]) & (row['word_article']!=''):
                if (row['word_article'][-1]!='s'):
                    intents.append(template[0] + ' ' + row['word_article']+ ' ' +row[col]+template[1])
                else:
                    intents.append(template[0] +' n' + ' ' + row['word_article']+' '+row[col]+template[1])            

            elif ('a'==template[0][-1:]) & (row['word_article']!=''):
                if (row['word_article'][-1]!='s'):
                    intents.append(template[0] + ' ' + row['word_article']+ ' ' +row[col]+template[1])
                else:
                    intents.append(template[0] + 'n' + ' ' + row['word_article']+' '+row[col]+template[1])                 
            else:
                intents.append(template[0] + ' ' +row[col]+template[1])                
        
            # Generating a pandas dataframe with the information needed to train a Rasa Bot
        df.loc[:,'intent_'+str(i+1)] = ''
        df.loc[:,'intent_'+str(i+1)] = intents

    # Finishing the dataset generation
    for colu in df.columns:
        df.loc[:, colu] = df[colu].str.capitalize()
    df.drop(['word_article', 'contexto'], axis=1, inplace=True)

    return df




def HuggingFace_Dataset(df, col, Question_Template):

    # Determining the grammatical articles to use for the input concepts
    df = Finding_GrammarArticles(df, col)

    # Creating the concepts questions
    for i, template in enumerate([Question_Template]):

        # Reviewing grammatical rules for the use of the intent templates
        questions = []
        for index, row in df.iterrows():

            
            if ('es'==template[0][-2:]) & (row['word_article']!=''):
                if (row['word_article'][-1]!='s'):
                    questions.append(template[0] + ' ' + row['word_article']+ ' ' +row[col]+template[1])
                else:
                    questions.append(template[0][:-2] + 'son' + ' ' + row['word_article']+' '+row[col]+template[1])

            elif ('ve'==template[0][-2:]) & (row['word_article']!=''):
                if (row['word_article'][-1]!='s'):
                    questions.append(template[0] + ' ' + row['word_article']+ ' ' +row[col]+template[1])
                else:
                    questions.append(template[0][:-5] + 'sirven' + ' ' + row['word_article']+' '+row[col]+template[1])

            elif ('ca'==template[0][-2:]) & (row['word_article']!=''):
                if (row['word_article'][-1]!='s'):
                    questions.append(template[0] + ' ' + row['word_article']+ ' ' +row[col]+template[1])
                else:
                    questions.append(template[0] +' n' + ' ' + row['word_article']+' '+row[col]+template[1])            

            elif ('a'==template[0][-1:]) & (row['word_article']!=''):
                if (row['word_article'][-1]!='s'):
                    questions.append(template[0] + ' ' + row['word_article']+ ' ' +row[col]+template[1])
                else:
                    questions.append(template[0] + 'n' + ' ' + row['word_article']+' '+row[col]+template[1])                 
            else:
                questions.append(template[0] + ' ' +row[col]+template[1])                
        
            # Generating a pandas dataframe with the information needed to train a Rasa Bot
        df.loc[:,col] = questions
        df = df.rename(columns={col:'pregunta', df.columns[2]:'respuesta', df.columns[1]:'contexto'})
        df = df[['pregunta', 'contexto', 'respuesta']]

    # Finishing the dataset generation
    for colu in df.columns:
        df.loc[:, colu] = df[colu].str.capitalize()


    return df
        
    
def Building_RobustDataset(df, templates, testper):

    '''
    Function that generates a robust dataset composed of inquiries and responses, where 
    the inquiries are generated using the input templates. It keeps unduplicated a
    percentage of all possible inquiries to make, this small dataset can be used for
    testing purposes. 
    
    inputs:

    df -> pandas dataframe containing at least two columns
    col_des -> Columns containing the concepts to be used for the intents generation
    Intents_Templates -> Templates for the generation of intents

    outputs:

    df -> Pandas dataframe containing information needed to train a Rasa bot

    '''
    
    test_idxs = df.sample(int(df.shape[0]*.1)).index
    df.loc[:, 'test'] = 0

    
    for i, typeofq in enumerate(templates):

        if i == 0:
            df_hug = HuggingFace_Dataset(df, 'concepto', (typeofq[0], typeofq[1]))
            df_hug['respuesta'] = i
            df_hug.loc[:, 'set_type'] = 0
            df_hug.loc[test_idxs, 'set_type'] = 1
        else:
            df_tmp = HuggingFace_Dataset(df.loc[~df.index.isin(test_idxs)], 'concepto', (typeofq[0], typeofq[1]))
            df_tmp['respuesta'] = i
            df_tmp.loc[:, 'set_type'] = 0
            df_hug = pd.concat([df_hug, df_tmp], ignore_index=True, sort=False)
        
    return df_hug




def Generating_Intents_Questions(concepto, concept, article, Question_Templates):

    '''
    Function to create question patterns for intents related to specific concepts.
        
    inputs:

    concept -> Concept base for the generation of question intent patterns
    article -> Grammatical article related to the conpcept input
    Question_Templates -> Questions templates to create the intent patterns

    outputs:

    questions -> list containing the generated question intent patterns

    '''

    # Creating the concepts questions
    questions = []
    for i, template in enumerate(Question_Templates):

        if ('es'==template[0][-2:]) & (article!=''):
            if (article[-1]!='s'):
                questions.append(template[0] + ' ' + article+ ' ' +concepto+template[1]+ '?')
                if concepto  != concept:
                    questions.append(template[0] + ' ' +concept+template[1]+ '?')
            else:
                questions.append(template[0][:-2] + 'son' + ' ' + article+' '+concepto+template[1]+ '?')
                if concepto  != concept:
                    questions.append(template[0] + ' ' + concept+template[1]+ '?')
    
        elif ('ve'==template[0][-2:]) & (article!=''):
            if (article[-1]!='s'):
                questions.append(template[0] + ' ' + article+ ' ' +concepto+template[1]+ '?')
                if concepto  != concept:
                    questions.append(template[0] + ' ' + concept+template[1]+ '?')
            else:
                questions.append(template[0][:-5] + 'sirven' + ' ' + article+' '+concepto+template[1]+ '?')
                if concepto  != concept:
                    questions.append(template[0] + ' ' + concept+template[1]+ '?')
    
        elif ('ca'==template[0][-2:]) & (article!=''):
            if (article[-1]!='s'):
                questions.append(template[0] + ' ' + article+ ' ' +concepto+template[1]+ '?')
                if concepto  != concept:
                    questions.append(template[0] + ' ' + concept+template[1]+ '?')
            else:
                questions.append(template[0] +' n' + ' ' + article+' '+concepto+template[1]+ '?')
                if concepto  != concept:
                    questions.append(template[0] + ' ' + concept+template[1]+ '?')           
    
        elif ('a'==template[0][-1:]) & (article!=''):
            if (article[-1]!='s'):
                questions.append(template[0] + ' ' + article+ ' ' +concepto+template[1] + '?')
                if concepto  != concept:
                    questions.append(template[0] + ' ' + concept+template[1]+ '?')
            else:
                questions.append(template[0] + 'n' + ' ' + article+' '+concepto+template[1]+ '?')
                if concepto  != concept:
                    questions.append(template[0] + ' ' + concept+template[1]+ '?')
        else:
            questions.append(template[0] + ' ' +concepto+template[1]+ '?')
            if concepto  != concept:
                questions.append(template[0] + ' ' + concept+template[1]+ '?')
        
    return questions






