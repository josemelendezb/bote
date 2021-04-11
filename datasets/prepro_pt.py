import pandas as pd

# Verifica que el documento sea consistente con el formato requerido
def validate_doc(lines):
  # Determinamos donde inicia y termina cada bloque [#Livro ... #Corpo]
  livro = []
  corpo = []
  for i, l in enumerate(lines):
    if l.find("#Livro") > -1: livro.append(i)
    if l.find("#Corpo") > -1: corpo.append(i)

  #Confirmamos que hay el mismo numero de etiquetas de apertura y cierre
  assert len(corpo) == len(livro)

  # Confirmamos que la apertuda esta antes del cierre
  for li, co in zip(livro, corpo):
    assert li < co

# Elimina las lineas que no correspondan al contenido de una review
def delete_trash_lines(lines):
  lines_clean = []
  discard = 0
  for i, l in enumerate(lines): # Descartamos la primera linea
    if l.find("#Livro") > -1: discard = 1
    
    if discard == 0 and i > 0: lines_clean.append(l)
    
    if l.find("#Corpo") > -1: discard = 0
  
  return lines_clean

#Verificar que todas las reviews tengan al menos un aspecto y una opinion
def aspect_opinion_exists(ap_tokens, op_tokens):
  ap_exist = False
  op_exist = False

  for ap in ap_tokens:
    if ap.find("OBJ") > -1:
      ap_exist = True
      break

  for op in op_tokens:
    if op.find("op") > -1:
      op_exist = True
      break

  return ap_exist*op_exist

# Se asegura que toda review tenga al menos un aspecto y una opinion
def get_valid_reviews(lines_clean):
  # Converir columnas en filas
  sentences = {
      "word": [],
      "pos": [],
      "aspect": [],
      "opinion": [],
      "polarity": [],
      "help": [],
      'triplets': []  
  }

  word, pos, aspect, opinion, polarity, help = [], [], [], [], [], []
  for i, l in enumerate(lines_clean):
    if l == '\n':
      #Verificar que todas las reviews tengan al menos un aspecto y una opinion
      if aspect_opinion_exists(aspect, opinion) == 1:
        sentences["word"].append(word)
        sentences["pos"].append(pos)
        sentences["aspect"].append(aspect)
        sentences["opinion"].append(opinion)
        sentences["polarity"].append(polarity)
        sentences["help"].append(help)
        
      word, pos, aspect, opinion, polarity, help = [], [], [], [], [], []
    else:
      ele = l.split("\t")
      word.append(ele[0].replace("\"", "'")) # Eliminar las comillas dobles
      pos.append("")#(ele[1].replace("\"", "'"))
      aspect.append(ele[2])
      opinion.append(ele[3])
      polarity.append(ele[4])
      help.append(ele[5].replace("\n", ""))
  
  return sentences

# Extraer cuantos aspectos hay en la oracion
def extract_num_aspects(ap_tokens):
  aspects = []
  for k, ap in enumerate(ap_tokens):
    
    if ap.find("OBJ") > -1:
      if ap not in aspects:
        aspects.append(ap)
  return aspects

# Extraer los indiices de un aspecto
def aspect_indices(aspect_code, ap_list):
  initial = ap_list.index(aspect_code)
  for i in range(initial + 1, len(ap_list)):
    if ap_list[i] != aspect_code:
      return [initial, i-1]

# Extraer cuantos aspectos hay en la oracion
def extract_num_opinions(op_tokens):
  opinions = []
  for k, op in enumerate(op_tokens):
    
    if op.find("op") > -1:
      tks = op.split(";")
      for tk in tks:
        if tk not in opinions:
          opinions.append(tk)
  return opinions

def indices_opinion(opinion_code, op_list):
  indices = []
  indices_temp = []
  for k, op in enumerate(op_list):
    if op.find(opinion_code) > -1:
      indices_temp.append(k)

      if k == len(op_list)-1: #Si es el ulitmo elemento
        indices.append([indices_temp[0], indices_temp[-1]])
    else:
      if len(indices_temp) > 0:
        indices.append([indices_temp[0], indices_temp[-1]])

      indices_temp = []
  return indices

# Extrac aspectos y confirmar que aquellos que tienen mas de un aspecto tienen asignada al menos una opinion
def orphan_aspect(sentences):
  l = 0
  for i in range(len(sentences["aspect"])):
    aspects = extract_num_aspects(sentences["aspect"][i])

    if len(aspects) > 1:
      op_string = " ".join(sentences["opinion"][i])
      for ap in aspects:
        numeral = ap[-2:]
        if op_string.find("op" + numeral) == -1:
          l += 1
          print(l)
          print(aspects)
          print(sentences["word"][i])
          print(sentences["aspect"][i])
          print(sentences["opinion"][i])
          print("############################################")
          break

# Extrac aspectos y determinar sus indices y los indices de sus respectivas opiniones y triplets
def get_triplets(sentences):
  triplets_all = []
  for i in range(len(sentences["aspect"])):
    triplets = []
    aspects = extract_num_aspects(sentences["aspect"][i])
    opinions = extract_num_opinions(sentences["opinion"][i])
    
    for ap in aspects:
      ap_indices = aspect_indices(ap, sentences["aspect"][i])
      op_indices = [] ##
      numeral = ap[-2:]
      for op_code in opinions:       
        op = "op"+numeral
        if op_code.find(op) > -1:
          op_indices = indices_opinion(op_code, sentences["opinion"][i])

      # Armar los triplets
      for ind_op in op_indices:
        sign = sentences["opinion"][i][ind_op[0]][-1]
        triplets.append((ap_indices, ind_op, sent[sign]))
    
    triplets_all.append(triplets)
  
  return triplets_all

#Divide o dataset em um conjunto de treinamento, validação e teste
#df: DataFrame com os dados
#porcentagens para dividir o dataset
def split_dataset(df, path, porcentagens = {"train": 0.65, "validation": 0.1, "test": 0.25}):
  print("\nGerando os conjuntos de treinamento, validação e teste...")
  df = df.sample(frac=1, random_state=31).reset_index(drop=True)

  n = round(len(df)*porcentagens["train"]), round(len(df)*porcentagens["validation"]), round(len(df)*porcentagens["test"])

  train_df, val_df, test_df = df.iloc[:n[0], :], df.iloc[n[0]:n[0] + n[1], :], df.iloc[n[0] + n[1]:, :]

  train_df.to_csv(path + "train_reli.txt", sep = "|", header = False, index = False)
  val_df.to_csv(path + "dev_triplets.txt", sep = "|", header = False, index = False)
  test_df.to_csv(path + "test_triplets.txt", sep = "|", header = False, index = False)

  print(f"O conjuntos foram gerados de acordo às seguintes proporções: {porcentagens} o numero de reviews por conjunto são {n[0]}, {n[1]} e {n[2]} respectivamente.")


  if __name__ == '__main__':

    files = [
            'ReLi-Amado.txt',
            'ReLi-Meyer.txt',
            'ReLi-Orwell.txt',
            'ReLi-Reboucas.txt',
            'ReLi-Salinger.txt',
            'ReLi-Saramago.txt',
            'ReLi-Sheldon.txt',
    ]

    display_orphan_aspects = False
    sent = {"+": "POS", "-": "NEG"}
    content = []
    total_sentences = 0

    for fi in files:
    fin = open('ReLi/reli_raw/'+fi, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()

    validate_doc(lines)

    lines_clean = delete_trash_lines(lines)

    sentences = get_valid_reviews(lines_clean)

    num_sent = len(sentences['word'])
    total_sentences += num_sent
    print("Valid sentences in",fi,":", num_sent)

    if display_orphan_aspects:
        print("In", fi, " there are the next sentences with orphan aspects")
        orphan_aspect(sentences)

    sentences["triplets"] = get_triplets(sentences)

    #Consolidad todos los archivos
    for word, pos, aspect, opinion, polarity, help, triplets in zip(sentences["word"], sentences["pos"], sentences["aspect"], sentences["opinion"], sentences["polarity"], sentences["help"], sentences["triplets"]):
        content.append(" ".join(word) + "####" + str(triplets) + "####" + str(pos) + "####" + str(aspect) + "####" + str(opinion) + "####" + str(polarity) + "####" + str(help) + "####" + str(fi))

    # Save File
    data = pd.DataFrame(content)
    split_dataset(data, path = 'ReLi/')

    data.to_csv(folder + 'datasets/ReLi/total_triplets.txt', sep = "|", header = False, index = False)

    print("total sentences",total_sentences)