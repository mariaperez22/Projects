import json
from nltk.stem.snowball import SnowballStemmer
import os
import re
import sys
import math
from pathlib import Path
from typing import Optional, List, Union, Dict
import pickle

class SAR_Indexer:
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de artículos de Wikipedia
        
        Preparada para todas las ampliaciones:
          parentesis + multiples indices + posicionales + stemming + permuterm

    Se deben completar los metodos que se indica.
    Se pueden añadir nuevas variables y nuevos metodos
    Los metodos que se añadan se deberan documentar en el codigo y explicar en la memoria
    """

    # lista de campos, el booleano indica si se debe tokenizar el campo
    # NECESARIO PARA LA AMPLIACION MULTIFIELD
    fields = [
        ("all", True), ("title", True), ("summary", True), ("section-name", True), ('url', False),
    ]
    def_field = 'all'
    PAR_MARK = '%'
    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10

    all_atribs = ['urls', 'index', 'sindex', 'ptindex', 'docs', 'weight', 'articles',
                  'tokenizer', 'stemmer', 'show_all', 'use_stemming']

    def __init__(self):
        """
        Constructor de la classe SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA

        Incluye todas las variables necesaria para todas las ampliaciones.
        Puedes añadir más variables si las necesitas 

        """
        self.urls = set() # hash para las urls procesadas,
        self.index = {} # hash para el indice invertido de terminos --> clave: termino, valor: posting list
        self.sindex = {} # hash para el indice invertido de stems --> clave: stem, valor: lista con los terminos que tienen ese stem
        self.ptindex = {} # hash para el indice permuterm.
        self.docs = {} # diccionario de terminos --> clave: entero(docid),  valor: ruta del fichero.
        self.weight = {} # hash de terminos para el pesado, ranking de resultados.
        self.articles = {} # hash de articulos --> clave entero (artid), valor: la info necesaria para diferencia los artículos dentro de su fichero
        self.tokenizer = re.compile("\W+") # expresion regular para hacer la tokenizacion
        self.stemmer = SnowballStemmer('spanish') # stemmer en castellano
        self.show_all = False # valor por defecto, se cambia con self.set_showall()
        self.show_snippet = False # valor por defecto, se cambia con self.set_snippet()
        self.use_stemming = False # valor por defecto, se cambia con self.set_stemming()
        self.use_ranking = False  # valor por defecto, se cambia con self.set_ranking()


    ###############################
    ###                         ###
    ###      CONFIGURACION      ###
    ###                         ###
    ###############################


    def set_showall(self, v:bool):
        """

        Cambia el modo de mostrar los resultados.
        
        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_all es True se mostraran todos los resultados el lugar de un maximo de self.SHOW_MAX, no aplicable a la opcion -C

        """
        self.show_all = v


    def set_snippet(self, v:bool):
        """

        Cambia el modo de mostrar snippet.
        
        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_snippet es True se mostrara un snippet de cada noticia, no aplicable a la opcion -C

        """
        self.show_snippet = v


    def set_stemming(self, v:bool):
        """

        Cambia el modo de stemming por defecto.
        
        input: "v" booleano.

        UTIL PARA LA VERSION CON STEMMING

        si self.use_stemming es True las consultas se resolveran aplicando stemming por defecto.

        """
        self.use_stemming = v



    #############################################
    ###                                       ###
    ###      CARGA Y GUARDADO DEL INDICE      ###
    ###                                       ###
    #############################################


    def save_info(self, filename:str):
        """
        Guarda la información del índice en un fichero en formato binario
        
        """
        info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'wb') as fh:
            pickle.dump(info, fh)

    def load_info(self, filename:str):
        """
        Carga la información del índice desde un fichero en formato binario
        
        """
        #info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'rb') as fh:
            info = pickle.load(fh)
        atrs = info[0]
        for name, val in zip(atrs, info[1:]):
            setattr(self, name, val)

    ###############################
    ###                         ###
    ###   PARTE 1: INDEXACION   ###
    ###                         ###
    ###############################

    def already_in_index(self, article:Dict) -> bool:

        return article['url'] in self.urls


    def index_dir(self, root:str, **args):
        """
        
        Recorre recursivamente el directorio o fichero "root" 
        NECESARIO PARA TODAS LAS VERSIONES
        
        Recorre recursivamente el directorio "root"  y indexa su contenido
        los argumentos adicionales "**args" solo son necesarios para las funcionalidades ampliadas

        """
        self.multifield = args['multifield']
        self.positional = args['positional']
        self.stemming = args['stem']
        self.permuterm = args['permuterm']

        file_or_dir = Path(root)
        
        if file_or_dir.is_file():
            # is a file
            self.index_file(root)
        elif file_or_dir.is_dir():
            # is a directory
            for d, _, files in os.walk(root):
                for filename in files:
                    docid = len(self.docs)
                    self.docs[docid] = filename
                    if filename.endswith('.json'):
                        fullname = os.path.join(d, filename)
                        self.index_file(fullname)
        else:
            print(f"ERROR:{root} is not a file nor directory!", file=sys.stderr)
            sys.exit(-1)

        ##########################################
        ## COMPLETAR PARA FUNCIONALIDADES EXTRA ##
        ##########################################
        
        if self.stemming:
            self.make_stemming()
        if self.permuterm:
            self.make_permuterm()
        
        
    def parse_article(self, raw_line:str) -> Dict[str, str]:
        """
        Crea un diccionario a partir de una linea que representa un artículo del crawler

        Args:
            raw_line: una linea del fichero generado por el crawler

        Returns:
            Dict[str, str]: claves: 'url', 'title', 'summary', 'all', 'section-name'
        """
        
        article = json.loads(raw_line)
        sec_names = []
        txt_secs = ''
        for sec in article['sections']:
            txt_secs += sec['name'] + '\n' + sec['text'] + '\n'
            txt_secs += '\n'.join(subsec['name'] + '\n' + subsec['text'] + '\n' for subsec in sec['subsections']) + '\n\n'
            sec_names.append(sec['name'])
            sec_names.extend(subsec['name'] for subsec in sec['subsections'])
        article.pop('sections') # no la necesitamos 
        article['all'] = article['title'] + '\n\n' + article['summary'] + '\n\n' + txt_secs
        article['section-name'] = '\n'.join(sec_names)

        return article
                
    
    def index_file(self, filename:str):
        """

        Indexa el contenido de un fichero.
        
        input: "filename" es el nombre de un fichero generado por el Crawler cada línea es un objeto json
            con la información de un artículo de la Wikipedia

        NECESARIO PARA TODAS LAS VERSIONES

        dependiendo del valor de self.multifield y self.positional se debe ampliar el indexado

        j es un diccionario procesado en el que todos los valores son cadenas -> 
        las claves del diccionario son: url, title, summary, all, section-name
           

        """
        #
        # 
        # En la version basica solo se debe indexar el contenido "article"
        #
        #
        #
        #################
        ### COMPLETAR ###
        #################

        for i, line in enumerate(open(filename)):
            j = self.parse_article(line) 

            if j['url'] not in self.urls:
                artid = len(self.articles)
                self.articles[artid] = line
                self.urls.add(j['url'])
            # Según self.multifield (se indexan todos los campos) y self.positional (posicion de los tokens en el indice)
            if self.multifield:
                campos = ['title', 'summary', 'all', 'section-name']
            else:
                campos = ['all'] 

            for campo in campos:
                contenido = j[campo]
                tokens = self.tokenize(contenido)

                #si esta activa agregar informacion sobre la posicion#
                if self.positional: 
                    for position, token in enumerate(tokens):
                        if token not in self.index:
                            self.index[token] = {i: [position]}
                        else:
                            if i in self.index[token]:
                                self.index[token][i].append(position)
                            else:
                                self.index[token][i] = [position]


                #positional no activo -> no añade información de posición
                else:
                    for token in tokens:
                        if token not in self.index:
                            self.index[token] = [i]
                        elif self.index[token][-1] != i:
                            self.index[token].append(i)

    def set_stemming(self, v:bool):
        """

        Cambia el modo de stemming por defecto.
        
        input: "v" booleano.

        UTIL PARA LA VERSION CON STEMMING

        si self.use_stemming es True las consultas se resolveran aplicando stemming por defecto.

        """
        self.use_stemming = v


    def tokenize(self, text:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Tokeniza la cadena "texto" eliminando simbolos no alfanumericos y dividientola por espacios.
        Puedes utilizar la expresion regular 'self.tokenizer'.

        params: 'text': texto a tokenizar

        return: lista de tokens

        """
        return self.tokenizer.sub(' ', text.lower()).split()


    def make_stemming(self):
        """

        Crea el indice de stemming (self.sindex) para los terminos de todos los indices.

        NECESARIO PARA LA AMPLIACION DE STEMMING.

        "self.stemmer.stem(token) devuelve el stem del token"


        """
        for field, tok in self.fields: # Para cada campo
            if self.multifield or field == 'all': # En caso multifield
                fieldDict = self.index.keys() # Me da una posting list 
                self.sindex[field] = {} # Creamos el diccionario de stems para el campo
                fieldSindex = self.sindex[field]
                for word in fieldDict: # Para cada palabra el indice del campo
                    stemedWord = self.stemmer.stem(word)
                    # Sino tiene valor para esta clave lo creamos
                    fieldSindex[stemedWord] = fieldSindex.get(stemedWord, [])
                    fieldSindex[stemedWord].append(word) # Añadimos la palabra asociada al stem


    
    def make_permuterm(self):
        """
        NECESARIO PARA LA AMPLIACION DE PERMUTERM

        Crea el indice permuterm (self.ptindex) para los terminos de todos los indices.

        """
        
        for field, tok in self.fields:
            if self.multifield or field == 'all':
                terms = self.index.keys()
                self.ptindex[field] = []
                fieldPtindex = self.ptindex[field]
                #Genera permutaciones para cada palabra en el diccionario de términos
                for word in terms:
                    #Crea una lista de permutaciones para la palabra actual
                    permWords = [word[i:] + '$' + word[:i] for i in range(len(word) + 1)]
                    #Agrega las tuplas (permutación, palabra original) a fieldPtindex
                    fieldPtindex.extend([(permWord, word) for permWord in permWords])
                #Ordena fieldPtindex en orden lexicográfico
                fieldPtindex.sort()
        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################




    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        
        Muestra estadisticas de los indices
        
        """
        pass
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
                
        print("=" * 40)
        print("Number of indexed files:", len(self.docs))
        print("-" * 40)
        print("Number of indexed articles:", len(self.articles))
        print("-" * 40)
        print("TOKENS:")
        if self.multifield:
            print("     # of tokens in 'all':", self.weight['all']['ntokens'])
            print("     # of tokens in 'title':", self.weight['all']['ntokens'])
            print("     # of tokens in 'summary':", self.weight['all']['ntokens'])
            print("     # of tokens in 'section-name':", self.weight['all']['ntokens'])
            print("     # of tokens in 'url':", self.urls)
        else:
            print("     # of tokens in 'article':", len(self.index))
        print("-" * 40)
        if self.permuterm:
            print('PERMUTERMS:')
            for field in self.ptindex.keys():
                print("\t# of permuterms in '{}': {}".format(field, len(self.ptindex[field])))
            print("-" * 40)
        if self.stemming:
            print('STEMS:')
            for field in self.sindex.keys():
                print("\t# of stems in '{}': {}".format(field, len(self.sindex[field])))
            print("-" * 40)
        print('Positional queries are ' +
            ('' if self.positional else 'NOT ') + 'allowed.')
        print("=" * 40)

        



    #################################
    ###                           ###
    ###   PARTE 2: RECUPERACION   ###
    ###                           ###
    #################################

    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################


    def solve_query(self, query:str, prev:Dict={}):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una query.
        Debe realizar el parsing de consulta que sera mas o menos complicado en funcion de la ampliacion que se implementen


        param:  "query": cadena con la query
                "prev": incluido por si se quiere hacer una version recursiva. No es necesario utilizarlo.


        return: posting list con el resultado de la query

        """

        if query is None or len(query) == 0:
            return []

        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################




    def get_posting(self, term:str, field:Optional[str]=None):
        """

        Devuelve la posting list asociada a un termino. 
        Dependiendo de las ampliaciones implementadas "get_posting" puede llamar a:
            - self.get_positionals: para la ampliacion de posicionales
            - self.get_permuterm: para la ampliacion de permuterms
            - self.get_stemming: para la amplaicion de stemming


        param:  "term": termino del que se debe recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario si se hace la ampliacion de multiples indices

        return: posting list
        
        NECESARIO PARA TODAS LAS VERSIONES

        """
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        pass



    def get_positionals(self, terms:str, index):
        """

        Devuelve la posting list asociada a una secuencia de terminos consecutivos.
        NECESARIO PARA LA AMPLIACION DE POSICIONALES

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        pass
        ########################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE POSICIONALES ##
        ########################################################


    def get_stemming(self, term:str, field: Optional[str]=None):
        """

        Devuelve la posting list asociada al stem de un termino.
        NECESARIO PARA LA AMPLIACION DE STEMMING

        param:  "term": termino para recuperar la posting list de su stem.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        
        stem = self.stemmer.stem(term)
        wordList = self.sindex[field].get(stem, [])
        if len(wordList) == 0: return wordList # Si no existe palabra en el indice asociada al stem, devolvemos una lista vacia
        fieldDict = self.index[field]
        resPosting = fieldDict.get(wordList[0], [])
        for word in wordList[1:]:
            wordPosting = fieldDict.get(word, [])
            resPosting = self.or_posting(resPosting, wordPosting)
        return resPosting

    def get_permuterm(self, term:str, field:Optional[str]=None):
        """

        Devuelve la posting list asociada a un termino utilizando el indice permuterm.
        NECESARIO PARA LA AMPLIACION DE PERMUTERM

        param:  "term": termino para recuperar la posting list, "term" incluye un comodin (* o ?).
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """

        ##################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA PERMUTERM ##
        ##################################################
        lista = []
        #Determinar si el comodín utilizado es * o ?
        com = False
        if "*" in term:
            pos = term.find("*")
        else:
            com = True
            pos = term.find("?")
        #Obtener la longitud de la cadena
        long = len(term)    
        buscar = term[pos+1:] + "$" + term[:pos]
        #Obtener las claves del índice permuterm
        claves = self.ptindex[field].keys()
        #Buscar las palabras que encajan con la busqueda
        for clave in claves:
            if clave.startswith(buscar) or clave==buscar:
                #Obtener las palabras asociadas a esa clave
                palabras = self.ptindex[field][clave]
                for pal in palabras:
                    if pal not in lista:
                        if com:
                            if long == len(pal):
                                lista.append(pal)
                        else:
                            lista.append(pal)
        #Obtener las claves del índice original
        clavesindex = self.index[field].keys()
        postinglist=[]
        #Si la palabra está presente en las claves del índice original,obtiene la lista de publicación asociada la palabra
        for word in lista:
            if word in clavesindex:
                listapost = self.index[field][word]
                for p in listapost:
                    if p not in postinglist:
                        postinglist.append(p)
        return postinglist



    def reverse_posting(self, p:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.


        param:  "p": posting list


        return: posting list con todos los artid exceptos los contenidos en p

        """
        
        pass
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################



    def and_posting(self, p1:list, p2:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el AND de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos en p1 y p2

        """
        
        pass
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        
        result = []
        index_p1 = 0
        index_p2 = 0
        while index_p1<len(p1) and index_p2<len(p2): 
            
            if p1[index_p1] < p2[index_p2]:
                
                index_p1 = index_p1 + 1

            elif p1[index_p1] > p2[index_p2]:
                                
                index_p2 = index_p2 + 1
                
            else:

                result.append(p1[index_p1])
                index_p1 = index_p1 + 1
                index_p2 = index_p2 + 1
                
        return result



    def or_posting(self, p1:list, p2:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el OR de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos de p1 o p2

        """

        pass
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        
        result = []
        index_p1 = 0
        index_p2 = 0
        
        while index_p1<len(p1) and index_p2<len(p2):
            
            if p1[index_p1] < p2[index_p2]:                
                result.append(p1[index_p1])
                index_p1 = index_p1 + 1
                
            elif p1[index_p1] > p2[index_p2]:
                result.append(p2[index_p2])
                index_p2 = index_p2 + 1
            
            else:
                result.append(p1[index_p1])
                index_p1 = index_p1 + 1
                index_p2 = index_p2 + 1
        
        while index_p1<len(p1):
            result.append(p1[index_p1])
            index_p1 = index_p1 + 1
        
        while index_p2<len(p2):
            result.append(p2[index_p2])
            index_p2 = index_p2 + 1
        
        return result


    def minus_posting(self, p1, p2):
        """
        OPCIONAL PARA TODAS LAS VERSIONES

        Calcula el except de dos posting list de forma EFICIENTE.
        Esta funcion se incluye por si es util, no es necesario utilizarla.

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos de p1 y no en p2

        """

        
        pass
        ########################################################
        ## COMPLETAR PARA TODAS LAS VERSIONES SI ES NECESARIO ##
        ########################################################
        
        result = []
        index_p1 = 0
        index_p2 = 0
        
        while index_p1 < len(p1) and index_p2 < len(p2):
            if p1[index_p1] < p2[index_p2]:
                result.append(p1[index_p1])
                index_p1 += 1
            elif p1[index_p1] > p2[index_p2]:
                index_p2 += 1
            else:
                index_p1 += 1
                index_p2 += 1
        
        # Agregar los elementos restantes de p1 si los hay
        result.extend(p1[index_p1:])  
        
        return result





    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################

    def solve_and_count(self, ql:List[str], verbose:bool=True) -> List:
        results = []
        for query in ql:
            if len(query) > 0 and query[0] != '#':
                r = self.solve_query(query)
                results.append(len(r))
                if verbose:
                    print(f'{query}\t{len(r)}')
            else:
                results.append(0)
                if verbose:
                    print(query)
        return results


    def solve_and_test(self, ql:List[str]) -> bool:
        errors = False
        for line in ql:
            if len(line) > 0 and line[0] != '#':
                query, ref = line.split('\t')
                reference = int(ref)
                result = len(self.solve_query(query))
                if reference == result:
                    print(f'{query}\t{result}')
                else:
                    print(f'>>>>{query}\t{reference} != {result}<<<<')
                    errors = True                    
            else:
                print(query)
        return not errors


    def solve_and_show(self, query:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra junto al numero de resultados 

        param:  "query": query que se debe resolver.

        return: el numero de artículo recuperadas, para la opcion -T

        """
        pass
        ################
        ## COMPLETAR  ##
        ################