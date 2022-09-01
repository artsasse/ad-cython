import random
import math
from math import isclose
from collections import deque
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt 
import heapq

'''
Retorna uma VA exponencial com taxa lambda, passada como argumento
'''
def generate_exp(lambda_param):

  # Gera a variavel aleatoria uniforme no intervalo [0.0, 1.0)
  u0 = random.random()

  # Pega amostra da exponencial
  x0 = - math.log(u0) / lambda_param
  
  # Retorna a amostra
  return x0

"""### Distribuições

#### Distribuições Exponenciais

Gerando distribuições exponenciais com λs específicos:
"""

def exp02():
  return generate_exp(0.2)


def exp04():
  return generate_exp(0.4)


def exp06():
  return generate_exp(0.6)


def exp08():
  return generate_exp(0.8)


def exp09():
  return generate_exp(0.9)


def exp10():
  return generate_exp(1.0)


def exp001():
  return generate_exp(0.01)

"""#### Distribuições Determinísticas

Para facilitar a verificação da corretude do simulador:
"""

"""
Distribuição com apenas um valor: 2
"""
def every2sec():
  return 2.0


"""
Classe para gerar uma distribuição com 2 numeros alternados entre si
Ex: 1, 2, 1, 2, 1 ...
"""
class AlternateDist:
  def __init__(self,value1, value2):
    self.value1 = value1
    self.value2 = value2
    self.alternate = value1
  # Gera um valor e já alterna para a proxima amostra
  def sample(self):
    value = self.alternate
    if self.alternate == self.value1:
      self.alternate = self.value2
      return value
    else:
      self.alternate = self.value1
      return value

"""### Seeds

Os seeds em Python são definidos por meio da função `random.seed(int)` e fixamos um seed antes cada simulação para garantir a reprodutibilidade dos resultados. Para garantir a independência dos seeds, rodamos o algoritmo com alguns seeds diferentes ao longo do trabalho.

Além disso também fizemos alguns testes para mostrar que não há sobreposição entre os intervalos gerados pelos seeds.

Primeiro geramos 2 vetores com váriaveis exponenciais com mesma taxa, cada um a partir de uma seed diferente. E depois verificamos se existem valores em comuns nesses vetores.

No outro teste geramos 2 vetores também e verificamos o indíce de correlação (Pearson) entre eles.

Mesmo usando seeds próximas (0 e 1, por exemplo) não encontramos sobreposição das sequências de valores nem uma correlação significativa. Acreditamos que isso se deve ao fato de o Python utilizar o algoritmo do Mersenne Twister para gerar números pseudoaleatórios (https://en.wikipedia.org/wiki/Mersenne_Twister).
"""

'''
- Recebe: 
  - número de rodadas, tamanho da amostra, taxa lambda, seed
- Retorna:
  - uma lista de rodadas com números aleatorios gerados pela amostra exponencial
'''
def get_samples_rounds(number_rounds, number_samples, lambda_generic, seed):
  random.seed(seed)
  rounds=[]
  for x in range(number_rounds):
    samples=[]
    for y in range(number_samples):
      samples.append(generate_exp(lambda_generic))
    rounds.append(samples)
  return rounds

#Exemplo 1

'''
- Recebe:
  - lista A de rodadas gerada pela função get_samples_rounds
  - lista B de rodadas gerada pela função get_samples_rounds
- Retorna:
  - "There are equal values", caso tenha encontrado um valor em comum nas lista
  - "No equal values", caso as listas não possuem valores em comum
'''
def simple_test(roundsA, roundsB):
  size_rounds_list = len(roundsA)
  for x in range(size_rounds_list-1):
    list_turn = roundsA[x]
    for y in range(size_rounds_list):
      list_verify = roundsB[y]
      result = list(set(list_turn).intersection(list_verify))
      if len(result) > 0:
        return "There are equal values"
  return "No equal values" 

# Exemplo 2
'''
- Recebe:
  - número de amostras de cada lista. A função gera apenas duas listas para 
    testar a correlação
- Retorna:
  - o valor da correlação de Pearson
'''
def test_pearson(n_samples, seed1, seed2):
  
  rounds = []  
  samples1 = []
  samples2 = []

  random.seed(seed1)
  for x in range(n_samples):
    samples1.append(random.random())
  rounds.append(samples1)
  random.seed(seed2)
  for x in range(n_samples):
    samples2.append(random.random())
  rounds.append(samples2)

  x = pd.Series(rounds[0])
  y = pd.Series(rounds[1])
  result = x.corr(y)
  return result

"""### Constantes

Algumas constante utilizadas para facilitar o entendimento do código:
"""

ARRIVAL = 1
END_OF_SERVICE = 0
MU = 1.0
IDLE = 0
BUSY = 1
FIFO = 0
LIFO = 1

"""### Eventos e Clientes"""

"""
Representam os eventos de chegada ou fim de serviço
"""
class Event:
  # Construtor da classe Evento
  def __init__(self, event_type, t, client_id):
    # Chegada ou fim de serviço
    self.event_type = event_type
    # Instante que o evento ocorre
    self.t = t
    # ID do cliente ao qual o evento esta relacionado
    self.client_id = client_id
    # Define como a classe vai ser impressa
  def __repr__(self) -> str:
    if self.event_type == ARRIVAL:
      type_str = "ARRIVAL"
    else:
      type_str = "END_OF_SERVICE"
    return f"event_type: {type_str}\nt: {self.t}\nclient_id: {self.client_id}"

"""
O cliente que chega e é servido.
Serve para guardar o tempo de espera até o serviço.
"""
class Client:
  def __init__(self, arrival_event, color):
    self.id = arrival_event.client_id
    self.arrival_time = arrival_event.t
    # tempo de espera até ser servido, deve ser alterado quando for atendido
    self.color = color
    self.waiting_time = 0.0
  # Define como a classe vai ser impressa
  def __repr__(self) -> str:
    return f"id: {self.id}\narrival_time: {self.arrival_time}\nwaiting_time: {self.waiting_time}"


def create_client(event, color):
  # Cria uma instancia da classe Cliente
  new_client = Client(event, color)
  # Retorna referencia para o cliente criado
  return new_client

"""### Gerar chegadas e fins de serviços"""

'''
- Recebe o t(tempo atual), arrival_dist (função que gera uma amostragem da 
distribuição desejada para a chegada) e o client_id (id do cliente que chegou)
- Retorna o evento gerado de chegada
'''
def generate_arrival(t, arrival_dist, client_id):
  arrival = Event(ARRIVAL, t + arrival_dist(), client_id)
  return arrival


'''
- Recebe o t(tempo atual), arrival_dist (função que gera uma amostragem da 
distribuição desejada para o tempo de serviço) e o client_id (id do cliente que
vai ser servido)
- Retorna o evento gerado de fim de serviço
'''
def generate_end_of_service(t, service_dist, client_id):
  end_of_service = Event(END_OF_SERVICE, t + service_dist(), client_id)
  return end_of_service

# TESTES
'''
Função auxiliar para imprimir os atributos de um evento
'''
def print_event(event, event_name):
  print(f"{event_name}.client_id: {event.client_id}")
  print(f"{event_name}.t: {event.t}")
  print(f"{event_name}.event_type: {event.event_type}")

# Chegada
arrival = generate_arrival(1, partial(generate_exp, 0.3), 0)
print_event(arrival, "arrival")
print("")

# Fim de serviço
end_of_service = generate_end_of_service(10, partial(generate_exp, 0.6), 1)
print_event(end_of_service, "end_of_service")

"""### Lista de Eventos

Organizamos a lista de eventos numa heap, levando em conta o instante de cada evento como critério para orderná-los e o tipo de evento como critério de desempate (fins de serviço vêm antes de chegadas). O terceiro critério de desempate é o id de cada evento. O evento com o menor id (ou seja, foi criado antes) acontece primeiro na simulação.
"""

class EventsList:
  def __init__(self):
    self.events = []
  def add(self, event):
    # Temos que adicionar o tempo e o id para critério de comparação para a heap
    heapq.heappush(self.events, (event.t, event.event_type, event.client_id, event))
  def pop(self):
    # Pega o evento que ocorre mais cedo, já que a lista esta ordenada pelos tempos
    t, event_type, id, event = heapq.heappop(self.events)
    return event


def get_event_time(event):
  return event.t


"""### Fila de Clientes"""

class ClientsQueue:
  def __init__(self):
    self.clients = deque([])
  def add(self, client):
    self.clients.append(client)
  def pop(self, queue_discipline):
    if queue_discipline == FIFO:
      return self.clients.popleft()  # Se for FIFO tira do inicio da lista
    else:
      return self.clients.pop()  # Se for LIFO tira do fim da lista


"""### Serviço"""

"""
client -> Cliente a ser servido
t -> Instante atual do sistema
service_dist -> distribuição do tempo de serviço
events_list -> lista de eventos para adicionar o fim do serviço
"""
def serve_client(client, t, service_dist, events_list):
  # Registra o tempo de espera do serviço
  client.waiting_time = t - client.arrival_time
  # Cria evento do fim do serviço
  end_of_service_event = generate_end_of_service(t, service_dist, client.id)
  # Adiciona evento do fim do servico a lista de eventos
  events_list.add(end_of_service_event)


"""### Intervalos de Confiança

#### t-Student

Como o número de amostras é sempre 3200 pelo enunciado e esse é um valor grande o bastante para t-Student se aproximar da Normal Unitária, usamos o valor 1.960 como o 100(1 - α/2)% percentil, quando α = 0.05 (seguindo a tabela da página 130 da apostila)
"""

T_PERCENTILE = 1.960
NSAMPLES = 3200
# Intervalo de confiança da t-student
def tstudent_ci(mean, variance):
  # Verifica se a media e variancia tem valores nulos (ou bem proximos disso)
  if isclose(mean, 0, abs_tol=1e-9):
    mean = 0.0
  if isclose(variance, 0, abs_tol=1e-9):
    variance = 0.0
  # Se variancia e media forem nulas
  if not(mean or variance):
    return 0.0, 0.0, 0.0
  # desvio padrao (raiz quadrada da variancia)
  std_dev = math.sqrt(variance)
  # metade do intervalo
  term = T_PERCENTILE * (std_dev/math.sqrt(NSAMPLES))
  # limite superior do intervalo
  upper_limit = mean + term
  # limite inferior do intervalo
  lower_limit = mean - term
  # precisao do intervalo de confiança (metade do intervalo / centro do intervalo)
  precision = term / mean 
  return upper_limit, lower_limit, precision

"""#### Chi-Square

Com 3200 amostras o 100(1 - α/2)% percentil da Chi-Square é igual a 3357.658 e o 100(α/2)% percentil é 3044.13, quando α = 0.05, de acordo com dados coletados da calculadora online presente em https://stattrek.com/online-calculator/chi-square e diretamente pela linguagem de programação R (https://rdrr.io/snippets/).
"""

CHI_0975_PERCENTILE = 3357.658
CHI_0025_PERCENTILE = 3044.13

'''
- Recebe a variância estimada, o número de coletas por rodada da simulação e o
número de rodadas da simulação.
- Retorna o intervalo de confiança da variância
'''
def chi_square(variance, k, n):
  # Se a variancia é zero, nao há intervalo, é apenas zero
  if variance == 0.0:
    upper_limit, lower_limit, precision = 0.0, 0.0, 0.0
  else:
    upper_limit = (n - 1) * variance * k / CHI_0025_PERCENTILE
    lower_limit = (n - 1) * variance * k / CHI_0975_PERCENTILE
    # Aproximadamente 0.05, já que esse foi o motivo para escolher 3200 rodadas
    precision = (CHI_0975_PERCENTILE - CHI_0025_PERCENTILE) / (CHI_0975_PERCENTILE + CHI_0025_PERCENTILE )
  return upper_limit, lower_limit, precision

"""### Estimadores

3 classes de estimadores foram criadas.

A primeira, mais geral permite o cálculo iterativo da média e da variância de uma variável, recebendo uma amostra de cada vez.
"""

'''
Classe para calcular os estimadores de média e variancia de 
do tempo de espera na fila e numero medio de pessoas na fila de espera
'''
class Estimator:
  def __init__(self):
    # Soma das amostras
    self.samples_sum = 0.0
    # Soma dos quadrados das amostras
    self.squares_sum = 0.0
    # Numero de amostras
    self.n = 0
  def add_sample(self, sample):
    self.samples_sum += sample
    self.squares_sum += (sample**2)
    self.n += 1
  def mean(self):
    return self.samples_sum / self.n
  def variance(self):
    term1 = self.squares_sum / (self.n - 1)
    term2 = (self.samples_sum**2) / (self.n * (self.n - 1))
    return term1 - term2
  def tstudent_ci(self):
    return tstudent_ci(self.mean(), self.variance())
  def chi_square_var(self, round_size, rounds):
    return chi_square(self.variance(), round_size, rounds)

"""O segundo estimador utiliza contas específicas para facilitar o cálculo do número médio de pessoas a partir da área do gráfico clientes X tempo."""

"""
Classe para calcular o número medio de pessoas na fila de espera
durante uma rodada de simulação usando a área do gráfico clientes x tempo.
"""
class NqueueAreaEstimator:
  def __init__(self):
    # Soma das areas
    self.nqueue_area_sum = 0.0
    # Soma dos intervalos de tempo
    self.dt_sum = 0.0
  def add_sample(self, nqueue, dt):
    self.nqueue_area_sum += nqueue * dt
    self.dt_sum += dt
  def mean(self):
    return self.nqueue_area_sum / self.dt_sum

"""O terceiro estimador também utiliza cálculos específicos, mas focado em computar a variância do número de pessoas na fila de espera a partir da *pmf* desse valor."""

"""
Classe para calcular a variancia do numero de pessoas na fila de espera
durante uma rodada de simulação usando a pmf.
"""
class NqueuePmfEstimator:
  def __init__(self):
    # Soma dos quadrados dos Nqis coletados vezes o intervalo de tempo
    self.nqueue_squares_sum = 0.0
    # Soma dos Nqis coletados vezes o intervalo de tempo
    self.nqueue_sum = 0.0
    # Soma dos intervalos de tempo
    self.dt_sum = 0.0
  def add_sample(self, nqueue, dt):
    self.nqueue_squares_sum += (nqueue**2) * dt
    self.nqueue_sum += nqueue * dt
    self.dt_sum += dt
  def variance(self):
    second_moment = self.nqueue_squares_sum / self.dt_sum
    first_moment = self.nqueue_sum / self.dt_sum
    return second_moment - (first_moment**2)

"""### Simulação

A função principal deste trabalho.

Simula a chegada e saída de pessoas, utilizando a lista de eventos e uma fila para os fregueses (FCFS ou LCFS). Aqui são coletados os valores de tempo de espera e tamanho da fila de espera que serão utilizados para o cálculo dos estimadores.
"""

'''
- Recebe o número de rodadas, tamanho da rodada, disciplina da fila de 
fregueses, distribuição do tempo entre chegadas e distribuição do tempo de serviço
- Retorna vazio
- arrival_dist gera o tempo para uma chegada
- service_dist gera o tempo para o fim do serviço
'''
def simulate(rounds, round_size, queue_discipline, arrival_dist, service_dist,
             print_values=False, print_graph=False, graph_values=False, decimals=3,
             transient_rounds=0, debug=False):
  # tempo inicial
  t = 0.0
  # numero de pessoas na fila de espera
  nqueue = 0
  # Cliente no servidor
  client_in_server = None
  # ID inicial dos clientes
  client_id = 0
  # Cria o evento inicial
  first_arrival = generate_arrival(t, arrival_dist, client_id)
  # Incrementa o identificador de clientes
  client_id += 1
  # Cria a lista de eventos, com o evento inicial
  events_list = EventsList()
  events_list.add(first_arrival)
  # Cria a fila de espera de clientes, vazia
  clients_queue = ClientsQueue()
  # Estimadores de tempo de espera e tamanho da fila
  time_est = Estimator()
  nqueue_est = Estimator()
  # Estimadores da variancia pela t-student
  time_var_est = Estimator()
  nqueue_var_est = Estimator()
  # lista com os tempos de espera nas filas de cada rodada
  roundW = []
  # lista com os valores do estimador de E[W] ao longo do tempo
  W = []
  # lista com numeros de individuos na fila em cada rodada
  Nq = []
  # Loop das Rodadas
  for i in range(rounds + transient_rounds):
    # Numero de clientes servidos na rodada
    served_clients = 0
    # Definindo a cor da rodada
    color = "%06x" % random.randint(0, 0xFFFFFF)
    # Inicia os estimadores de tempo de espera e tamanho da fila da rodada
    round_time_est = Estimator()
    round_nqueue_est = NqueueAreaEstimator()
    round_nqueue_pmf_est = NqueuePmfEstimator()
    while(served_clients < round_size):
      # Seleciona o próximo evento e tira da lista
      event = events_list.pop()
      # No modo debug, imprime os eventos de cada rodada
      if debug:
        print(f"Round {i}: {event}")
        print("-----------")
      # Salva o valor de Nq durante o intervalo de tempo ate o evento
      round_nqueue_est.add_sample(nqueue, event.t - t)
      round_nqueue_pmf_est.add_sample(nqueue, event.t - t)
      # Avança o tempo para o instante do evento
      t = event.t
      # Se o evento é uma CHEGADA
      if event.event_type == ARRIVAL:
        # Cria um cliente
        new_client = create_client(event, color)
        # Se o servidor esta LIVRE
        if not client_in_server:
          # Coloca o cliente no servidor
          client_in_server = new_client
          # Registra tempo de espera e adiciona evento do fim do serviço na lista
          serve_client(new_client, t, service_dist, events_list)
        # Se o servidor esta OCUPADO
        else:
          # Adiciona o cliente na fila de espera
          clients_queue.add(new_client)
          # Incrementa numero de clientes na fila de espera
          nqueue += 1
        # Incrementa id de cliente
        client_id += 1
        # Cria nova chegada 
        new_arrival = generate_arrival(t, arrival_dist, client_id)
        # Adiciona nova chegada na lista de eventos
        events_list.add(new_arrival)
      # Se o evento é o FIM DE SERVIÇO
      else:
        # Se o cliente tiver a cor da rodada atual
        if color == client_in_server.color:
          # Salva o tempo de espera do cliente que acabou o serviço
          round_time_est.add_sample(client_in_server.waiting_time)
          # Incrementa numero de clientes que ja foram servidos na rodada
          served_clients += 1
        # Se nao ha NINGUEM na fila de espera
        if nqueue == 0:
          # Muda status do servidor para vazio
          client_in_server = None
        # Se ha ALGUEM na fila de espera
        else:
          # Pega o proximo cliente
          client = clients_queue.pop(queue_discipline)
          # Coloca o cliente no servidor
          client_in_server = client
          # Registra tempo de espera e adiciona evento do fim do serviço na lista
          serve_client(client, t, service_dist, events_list)
          # Decrementa numero de clientes na fila de espera
          nqueue -= 1
    # Se ja passou da fase transiente
    if i >= transient_rounds:
      # Calcula a estatisticas de W e Nq da rodada (amostras da rodada)
      round_time_mean = round_time_est.mean()
      round_time_var = round_time_est.variance()
      round_nqueue_mean = round_nqueue_est.mean()
      round_nqueue_var = round_nqueue_pmf_est.variance()
      # Salva as amostras da rodada nos estimadores gerais
      time_est.add_sample(round_time_mean)
      time_var_est.add_sample(round_time_var)
      nqueue_est.add_sample(round_nqueue_mean)
      nqueue_var_est.add_sample(round_nqueue_var)
    # Salva amostras para o grafico, se desejado
    if graph_values:
      W.append(time_est.mean())
      Nq.append(nqueue_est.mean())

  # Obtem os intervalos de confiança e seus centros
  # t-student para media do tempo
  w_mean_upper, w_mean_lower, w_mean_precision = time_est.tstudent_ci()
  w_mean_center = time_est.mean()
  # chi-square para variancia do tempo
  w_var_upper, w_var_lower, w_var_precision = time_est.chi_square_var(round_size, rounds)
  w_var_center = time_est.variance()
  # t-student para variancia do tempo
  w_var_upper2, w_var_lower2, w_var_precision2 = time_var_est.tstudent_ci()
  w_var_center2 = time_var_est.mean()
  # t-student para media de Nq
  nq_mean_upper, nq_mean_lower, nq_mean_precision = nqueue_est.tstudent_ci()
  nq_mean_center = nqueue_est.mean()
  # chi-square para variancia de Nq
  nq_var_upper, nq_var_lower, nq_var_precision = nqueue_est.chi_square_var(round_size, rounds)
  nq_var_center = nqueue_est.variance()
  # t-studente para variancia de Nq
  nq_var_upper2, nq_var_lower2, nq_var_precision2 = nqueue_var_est.tstudent_ci()
  nq_var_center2 = nqueue_var_est.mean()

  if print_values:
    print("(a) Média de W")
    print(f"Intervalo de Confiança: {w_mean_lower:.{decimals}f} a {w_mean_upper:.{decimals}f}")
    print(f"Precisão: {w_mean_precision * 100:.2f}%")
    print("------------------")
    print("(b) Variância de W - Chi-Square")
    print(f"Intervalo de Confiança: {w_var_lower:.{decimals}f} a {w_var_upper:.{decimals}f}")
    print(f"Precisão: {w_var_precision * 100:.2f}%")
    print("------------------")
    print("(b) Variância de W - t-Student")
    print(f"Intervalo de Confiança: {w_var_lower2:.{decimals}f} a {w_var_upper2:.{decimals}f}")
    print(f"Precisão: {w_var_precision2 * 100:.2f}%")
    print("------------------")
    print("(c) Média de Nq")
    print(f"Intervalo de Confiança: {nq_mean_lower:.{decimals}f} a {nq_mean_upper:.{decimals}f}")
    print(f"Precisão: {nq_mean_precision * 100:.2f}%")
    print("------------------")
    print("(d) Variância de Nq - Chi-Square")
    print(f"Intervalo de Confiança: {nq_var_lower:.{decimals}f} a {nq_var_upper:.{decimals}f}")
    print(f"Precisão: {nq_var_precision * 100:.2f}%")
    print("------------------")
    print("(d) Variância de Nq - t-Student")
    print(f"Intervalo de Confiança: {nq_var_lower2:.{decimals}f} a {nq_var_upper2:.{decimals}f}")
    print(f"Precisão: {nq_var_precision2 * 100:.2f}%")
  else:
    if queue_discipline == FIFO:
      discipline = "FCFS"
    else:
      discipline = "LCFS"
    return {"time_mean": [discipline, w_mean_lower, w_mean_upper, w_mean_precision, w_mean_center],
            "time_var_chi": [discipline, w_var_lower, w_var_upper, w_var_precision, w_var_center],
            "time_var_t": [discipline, w_var_lower2, w_var_upper2, w_var_precision2, w_var_center2],
            "nqueue_mean": [discipline, nq_mean_lower, nq_mean_upper, nq_mean_precision, nq_mean_center],
            "nqueue_var_chi": [discipline, nq_var_lower, nq_var_upper, nq_var_precision, nq_var_center],
            "nqueue_var_t": [discipline, nq_var_lower2, nq_var_upper2, nq_var_precision2, nq_var_center2],
            "time_graph_values": W,
            "nqueue_graph_values": Nq}



# Serviço a cada 1.5 segundo
def every1_5sec():
  return 1.5

rhos = [0.2, 0.4, 0.6, 0.8, 0.9]

def analytic_wait_time_mean(rho):
  return rho / (1 - rho)

def analytic_wait_time_var_fcfs(rho):
  term1 = rho / (1 - rho)
  term2 = term1 + 2
  return term1 * term2

def analytic_wait_time_var_lcfs(rho):
  term1 = rho / ((1 - rho)**2)
  term2 = (2 / (1 - rho)) - rho
  return term1 * term2

def analytic_nqueue_mean(rho):
  return (rho**2) / (1 - rho)

def analytic_nqueue_var(rho):
  term1 = rho / ((1 - rho)**2)
  return term1 - rho - (rho**2)

def get_analytic_values():
  
  utilizations = [0.2, 0.4, 0.6, 0.8, 0.9]
  # Pegar valores analiticos
  w_mean = []
  w_var_fcfs = []
  w_var_lcfs = []
  nq_mean = []
  nq_var = []
  for ut in utilizations:
    w_mean.append(analytic_wait_time_mean(ut))
    w_var_fcfs.append(analytic_wait_time_var_fcfs(ut))
    w_var_lcfs.append(analytic_wait_time_var_lcfs(ut))
    nq_mean.append(analytic_nqueue_mean(ut))
    nq_var.append(analytic_nqueue_var(ut))

  return w_mean, w_var_fcfs, w_var_lcfs, nq_mean, nq_var

an_w_mean, an_w_var_fcfs, an_w_var_lcfs, an_nq_mean, an_nq_var = get_analytic_values()

"""## 5) Tabelas com os resultados e comentários pertinentes

### Função Geradora de Tabelas

### (a) Tempo Médio de Espera em Fila - E[W]

##### Função Geradora de Tabela
"""

# Testa a simulacao com a seed desejada e numero de coletas por rodada k_min
def get_time_mean_table(seed, k_min, queue_discipline=FIFO, print_graph=False):

  # Seed do gerador de numeros aleatorios
  random.seed(seed)

  # Cria a tabela de resultados
  df = pd.DataFrame(columns=["Utilization", "Discipline", "Lower Limit", "Upper Limit", "Precision", "Center", "Analytic Value"])

  # Lista de taxas de utilizacao possives
  exp_functions = [exp02, exp04, exp06, exp08, exp09]
  utilization_labels = ["0.2", "0.4", "0.6", "0.8", "0.9"]

  # Para cada indice e cada taxa de utilizacao
  for i, exp_function in enumerate(exp_functions):
    # Pega o intevalo de confianca da media do tempo de espera
    results_all = simulate(NSAMPLES, k_min, queue_discipline, exp_function, exp10, graph_values=print_graph)
    results = deque(results_all['time_mean'])
    if print_graph:
      plt.figure(num=1, figsize=(8,5), dpi=100)
      plt.plot(results_all['time_graph_values'][0], label=f"{'FIFO' if queue_discipline == FIFO else 'LIFO'} - ρ = {utilization_labels[i]}")
      plt.title("E[W]")
      plt.figure(num=2, figsize=(8,5), dpi=100)
      plt.plot(results_all['nqueue_graph_values'][0], label=f"{'FIFO' if queue_discipline == FIFO else 'LIFO'} - ρ = {utilization_labels[i]}")
      plt.title("E[Nq]")
      plt.figure(num=3, figsize=(8,5), dpi=100)
      plt.plot(results_all['time_graph_values'][1], label=f"{'FIFO' if queue_discipline == FIFO else 'LIFO'} - ρ = {utilization_labels[i]}")
      plt.title("var[W]")
      plt.figure(num=4, figsize=(8,5), dpi=100)
      plt.plot(results_all['nqueue_graph_values'][1], label=f"{'FIFO' if queue_discipline == FIFO else 'LIFO'} - ρ = {utilization_labels[i]}")
      plt.title("var[Nq]")
    # Adiciona as informacoes de utilizacao na lista
    results.appendleft(utilization_labels[i])
    # Adiciona valores analiticos
    results.append(an_w_mean[i])
    # Adiciona os resultados na tabela
    df.loc[len(df.index)] = results
  if print_graph:
    plt.figure(num=1, figsize=(8,5), dpi=100)
    plt.legend(loc='lower right')
    plt.figure(num=2, figsize=(8,5), dpi=100)
    plt.legend(loc='lower right')
    plt.figure(num=3, figsize=(8,5), dpi=100)
    plt.legend(loc='lower right')
    plt.figure(num=4, figsize=(8,5), dpi=100)
    plt.legend(loc='lower right')
    plt.show()
  return df


"""### (b) Variância do Tempo de Espera em Fila

##### Função Geradora de Tabela
"""

# Testa a simulacao com a seed desejada e numero de coletas por rodada k_min
def get_time_var_table(seed, k_min, queue_discipline=FIFO, ic="time_var_chi", print_graph=False):

  # Seed do gerador de numeros aleatorios
  random.seed(seed)

  # Cria a tabela de resultados
  df = pd.DataFrame(columns=["Utilization", "Discipline", "Lower Limit", "Upper Limit", "Precision", "Center", "Analytic Value"])

  # Lista de taxas de utilizacao possives
  exp_functions = [exp02, exp04, exp06, exp08, exp09]
  utilization_labels = ["0.2", "0.4", "0.6", "0.8", "0.9"]

  # Para cada indice e cada taxa de utilizacao
  for i, exp_function in enumerate(exp_functions):
    # Pega o intevalo de confianca da media do tempo de espera
    results_all = simulate(NSAMPLES, k_min, queue_discipline, exp_function, exp10)
    results = deque(results_all[ic])
    if print_graph:
      plt.figure(num=1, figsize=(8,5), dpi=100)
      plt.plot(results_all['time_graph_values'][0], label=f"{'FIFO' if queue_discipline == FIFO else 'LIFO'} - ρ = {utilization_labels[i]}")
      plt.title("E[W]")
      plt.figure(num=2, figsize=(8,5), dpi=100)
      plt.plot(results_all['nqueue_graph_values'][0], label=f"{'FIFO' if queue_discipline == FIFO else 'LIFO'} - ρ = {utilization_labels[i]}")
      plt.title("E[Nq]")
      plt.figure(num=3, figsize=(8,5), dpi=100)
      plt.plot(results_all['time_graph_values'][1], label=f"{'FIFO' if queue_discipline == FIFO else 'LIFO'} - ρ = {utilization_labels[i]}")
      plt.title("var[W]")
      plt.figure(num=4, figsize=(8,5), dpi=100)
      plt.plot(results_all['nqueue_graph_values'][1], label=f"{'FIFO' if queue_discipline == FIFO else 'LIFO'} - ρ = {utilization_labels[i]}")
      plt.title("var[Nq]")
    # Adiciona as informacoes de utilizacao na lista
    results.appendleft(utilization_labels[i])
    # Adiciona valores analiticos
    if queue_discipline == FIFO:
      results.append(an_w_var_fcfs[i])
    else:
      results.append(an_w_var_lcfs[i])
    # Adiciona os resultados na tabela
    df.loc[len(df.index)] = results
  if print_graph:
    plt.figure(num=1, figsize=(8,5), dpi=100)
    plt.legend(loc='lower right')
    plt.figure(num=2, figsize=(8,5), dpi=100)
    plt.legend(loc='lower right')
    plt.figure(num=3, figsize=(8,5), dpi=100)
    plt.legend(loc='lower right')
    plt.figure(num=4, figsize=(8,5), dpi=100)
    plt.legend(loc='lower right')
    plt.show()
  return df


"""### (c) Número Médio de Pessoas na Fila de Espera

##### Função Geradora de Tabela
"""

# Testa a simulacao com a seed desejada e numero de coletas por rodada k_min
def get_nqueue_mean_table(seed, k_min, queue_discipline=FIFO, print_graph=False):

  # Seed do gerador de numeros aleatorios
  random.seed(seed)

  # Cria a tabela de resultados
  df = pd.DataFrame(columns=["Utilization", "Discipline", "Lower Limit", "Upper Limit", "Precision", "Center", "Analytic Value"])

  # Lista de taxas de utilizacao possives
  exp_functions = [exp02, exp04, exp06, exp08, exp09]
  utilization_labels = ["0.2", "0.4", "0.6", "0.8", "0.9"]

  # Para cada indice e cada taxa de utilizacao
  for i, exp_function in enumerate(exp_functions):
    # Pega o intevalo de confianca da media do tempo de espera
    results_all = simulate(NSAMPLES, k_min, queue_discipline, exp_function, exp10)
    results = deque(results_all['nqueue_mean'])
    if print_graph:
      plt.figure(num=1, figsize=(8,5), dpi=100)
      plt.plot(results_all['time_graph_values'][0], label=f"{'FIFO' if queue_discipline == FIFO else 'LIFO'} - ρ = {utilization_labels[i]}")
      plt.title("E[W]")
      plt.figure(num=2, figsize=(8,5), dpi=100)
      plt.plot(results_all['nqueue_graph_values'][0], label=f"{'FIFO' if queue_discipline == FIFO else 'LIFO'} - ρ = {utilization_labels[i]}")
      plt.title("E[Nq]")
      plt.figure(num=3, figsize=(8,5), dpi=100)
      plt.plot(results_all['time_graph_values'][1], label=f"{'FIFO' if queue_discipline == FIFO else 'LIFO'} - ρ = {utilization_labels[i]}")
      plt.title("var[W]")
      plt.figure(num=4, figsize=(8,5), dpi=100)
      plt.plot(results_all['nqueue_graph_values'][1], label=f"{'FIFO' if queue_discipline == FIFO else 'LIFO'} - ρ = {utilization_labels[i]}")
      plt.title("var[Nq]")
    # Adiciona as informacoes de disciplina e utilizacao na lista
    results.appendleft(utilization_labels[i])
    # Valores analiticos
    results.append(an_nq_mean[i])
    '''
    if queue_discipline == FIFO:
      results.appendleft("FIFO")
    else:
      results.appendleft("LIFO")
    '''
    # Adiciona os resultados na tabela
    df.loc[len(df.index)] = results
  if print_graph:
    plt.figure(num=1, figsize=(8,5), dpi=100)
    plt.legend(loc='lower right')
    plt.figure(num=2, figsize=(8,5), dpi=100)
    plt.legend(loc='lower right')
    plt.figure(num=3, figsize=(8,5), dpi=100)
    plt.legend(loc='lower right')
    plt.figure(num=4, figsize=(8,5), dpi=100)
    plt.legend(loc='lower right')
    plt.show()
  return df

"""#### FCFS"""


"""### (d) Variância do Número de Pessoas na Fila de Espera

#### Função Geradora de Tabela
"""

# Testa a simulacao com a seed desejada e numero de coletas por rodada k_min
def get_nqueue_var_table(seed, k_min, queue_discipline=FIFO, ic="nqueue_var_chi", print_graph=False):

  # Seed do gerador de numeros aleatorios
  random.seed(seed)

  # Cria a tabela de resultados
  df = pd.DataFrame(columns=["Utilization", "Discipline", "Lower Limit", "Upper Limit", "Precision", "Center", "Analytic Value"])

  # Lista de taxas de utilizacao possives
  exp_functions = [exp02, exp04, exp06, exp08, exp09]
  utilization_labels = ["0.2", "0.4", "0.6", "0.8", "0.9"]

  # Para cada indice e cada taxa de utilizacao
  for i, exp_function in enumerate(exp_functions):
    # Pega o intevalo de confianca da media do tempo de espera
    results_all = simulate(NSAMPLES, k_min, queue_discipline, exp_function, exp10)
    results = deque(results_all[ic])
    if print_graph:
      plt.figure(num=1, figsize=(8,5), dpi=100)
      plt.plot(results_all['time_graph_values'][0], label=f"{'FIFO' if queue_discipline == FIFO else 'LIFO'} - ρ = {utilization_labels[i]}")
      plt.title("E[W]")
      plt.figure(num=2, figsize=(8,5), dpi=100)
      plt.plot(results_all['nqueue_graph_values'][0], label=f"{'FIFO' if queue_discipline == FIFO else 'LIFO'} - ρ = {utilization_labels[i]}")
      plt.title("E[Nq]")
      plt.figure(num=3, figsize=(8,5), dpi=100)
      plt.plot(results_all['time_graph_values'][1], label=f"{'FIFO' if queue_discipline == FIFO else 'LIFO'} - ρ = {utilization_labels[i]}")
      plt.title("var[W]")
      plt.figure(num=4, figsize=(8,5), dpi=100)
      plt.plot(results_all['nqueue_graph_values'][1], label=f"{'FIFO' if queue_discipline == FIFO else 'LIFO'} - ρ = {utilization_labels[i]}")
      plt.title("var[Nq]")
    # Adiciona as informacoes de utilizacao na lista
    results.appendleft(utilization_labels[i])
    # Valores analiticos
    results.append(an_nq_var[i])
    # Adiciona os resultados na tabela
    df.loc[len(df.index)] = results
  if print_graph:
    plt.figure(num=1, figsize=(8,5), dpi=100)
    plt.legend(loc='lower right')
    plt.figure(num=2, figsize=(8,5), dpi=100)
    plt.legend(loc='lower right')
    plt.figure(num=3, figsize=(8,5), dpi=100)
    plt.legend(loc='lower right')
    plt.figure(num=4, figsize=(8,5), dpi=100)
    plt.legend(loc='lower right')
    plt.show()
  return df



#retorna todas as tabelas da simulação
def get_all_tables(seed, k_min, utilization, queue_discipline=FIFO):

  # Seed do gerador de numeros aleatorios
  random.seed(seed)

  # Cria as tabelas de resultados
  df_time_mean = pd.DataFrame(columns=["Utilization", "Discipline", "Lower Limit", "Upper Limit", "Precision", "Center"])
  df_nqueue_mean = pd.DataFrame(columns=["Utilization", "Discipline", "Lower Limit", "Upper Limit", "Precision", "Center"])
  df_time_var_t = pd.DataFrame(columns=["Utilization", "Discipline", "Lower Limit", "Upper Limit", "Precision", "Center"])
  df_time_var_chi = pd.DataFrame(columns=["Utilization", "Discipline", "Lower Limit", "Upper Limit", "Precision", "Center"])
  df_nqueue_var_t = pd.DataFrame(columns=["Utilization", "Discipline", "Lower Limit", "Upper Limit", "Precision", "Center"])
  df_nqueue_var_chi = pd.DataFrame(columns=["Utilization", "Discipline", "Lower Limit", "Upper Limit", "Precision", "Center"])
  #executa a simulação
  results_all = simulate(NSAMPLES, k_min, queue_discipline, partial(generate_exp, utilization), exp10)
  
  #formata os resultados
  results_time_mean = deque(results_all['time_mean'])
  results_time_var_t = deque(results_all['time_var_t'])
  results_time_var_chi = deque(results_all['time_var_chi'])
  results_nqueue_mean = deque(results_all['nqueue_mean'])
  results_nqueue_var_chi = deque(results_all['nqueue_var_chi'])
  results_nqueue_var_t = deque(results_all['nqueue_var_chi'])

  #insere a utilização em todos os resultados
  results_time_mean.appendleft(str(utilization))
  results_time_var_t.appendleft(str(utilization))
  results_time_var_chi.appendleft(str(utilization))
  results_nqueue_mean.appendleft(str(utilization))
  results_nqueue_var_chi.appendleft(str(utilization))
  results_nqueue_var_t.appendleft(str(utilization))
  
  #insere os resultados no dataframe
  df_time_mean.loc[len(df_time_mean.index)] = results_time_mean
  df_time_var_t.loc[len(df_time_var_t.index)] = results_time_var_t
  df_time_var_chi.loc[len(df_time_var_chi.index)] = results_time_var_chi
  df_nqueue_mean.loc[len(df_nqueue_mean.index)] = results_nqueue_mean
  df_nqueue_var_chi.loc[len(df_nqueue_var_chi.index)] = results_nqueue_var_chi
  df_nqueue_var_t.loc[len(df_nqueue_var_t.index)] = results_nqueue_var_t

  return df_time_mean, df_time_var_t, df_time_var_chi, df_nqueue_mean, df_nqueue_var_chi, df_nqueue_var_t

"""#### Função unificada

"""

#retorna todas as tabelas da simulação
def get_one_table(seed, k_min, utilization, queue_discipline=FIFO):

  # Seed do gerador de numeros aleatorios
  random.seed(seed)

  # nomes das colunas
  columns = ["Utilization", "Discipline", "Lower Limit", "Upper Limit", "Precision", "Center"]

  # Cria as tabelas de resultados
  df_general = pd.DataFrame(columns=columns)

  #executa a simulação
  results_all = simulate(NSAMPLES, k_min, queue_discipline, partial(generate_exp, utilization), exp10)
  
  #formata os resultados
  results_time_mean = deque(results_all['time_mean'])
  results_time_var_t = deque(results_all['time_var_t'])
  results_time_var_chi = deque(results_all['time_var_chi'])
  results_nqueue_mean = deque(results_all['nqueue_mean'])
  results_nqueue_var_t = deque(results_all['nqueue_var_t'])
  results_nqueue_var_chi = deque(results_all['nqueue_var_chi'])

  #insere a utilização em todos os resultados
  results_time_mean.appendleft(str(utilization))
  results_time_var_t.appendleft(str(utilization))
  results_time_var_chi.appendleft(str(utilization))
  results_nqueue_mean.appendleft(str(utilization))
  results_nqueue_var_chi.appendleft(str(utilization))
  results_nqueue_var_t.appendleft(str(utilization))
  
  #insere os resultados no dataframe
  df_general.loc[len(df_general.index)] = results_time_mean
  df_general.loc[len(df_general.index)] = results_time_var_t
  df_general.loc[len(df_general.index)] = results_time_var_chi
  df_general.loc[len(df_general.index)] = results_nqueue_mean
  df_general.loc[len(df_general.index)] = results_nqueue_var_t
  df_general.loc[len(df_general.index)] = results_nqueue_var_chi

  # Nomeia as linhas
  df_general.index = ['E[W]', 'V[W] - t-Student', 'V[W] - Chi-Square', 'E[Nq]', 'V[Nq] - t-Student', 'V[Nq] - Chi-Square']

  # Arredonda os valores
  df_general = df_general.round(decimals = 3)

  return df_general


"""#### Main do projeto"""

def main():
    """Função principal da aplicação.
    """
    k_min = int(input("Please enter a integer for k_min(rounds_size):\n"))
    utilization = float(input("Please enter a float for utilization:\n"))
    seed = int(input("Please enter a int seed:\n"))
    
    print("Executing Simulation - FIFO\n")
    df_time_mean_fifo, df_time_var_t_fifo, df_time_var_chi_fifo, df_nqueue_mean_fifo, df_nqueue_var_chi_fifo, df_nqueue_var_t_fifo  = get_all_tables(seed, k_min, utilization, FIFO)
    
    print("Getting time csvs for FIFO\n")
    df_time_mean_fifo.to_csv('df_time_mean_fifo.csv', sep='\t', encoding='utf-8')
    df_time_var_t_fifo.to_csv('df_time_var_t_fifo.csv', sep='\t', encoding='utf-8')
    df_time_var_chi_fifo.to_csv('df_time_var_chi_fifo.csv', sep='\t', encoding='utf-8')
    
    print("Getting nqueue csvs for FIFO\n")
    df_nqueue_mean_fifo.to_csv('df_nqueue_mean_fifo.csv', sep='\t', encoding='utf-8')
    df_nqueue_var_chi_fifo.to_csv('df_nqueue_var_chi_fifo.csv', sep='\t', encoding='utf-8')
    df_nqueue_var_t_fifo.to_csv('df_nqueue_var_t_fifo.csv', sep='\t', encoding='utf-8')
    
    print("Executing Simulation - LIFO\n")
    df_time_mean_lifo, df_time_var_t_lifo, df_time_var_chi_lifo, df_nqueue_mean_lifo, df_nqueue_var_chi_lifo, df_nqueue_var_t_lifo  = get_all_tables(seed, k_min, utilization, LIFO)
    
    print("Getting time csvs for LIFO\n")
    df_time_mean_lifo.to_csv('df_time_mean_lifo.csv', sep='\t', encoding='utf-8')
    df_time_var_t_lifo.to_csv('df_time_var_t_lifo.csv', sep='\t', encoding='utf-8')
    df_time_var_chi_lifo.to_csv('df_time_var_chi_lifo.csv', sep='\t', encoding='utf-8')
    
    print("Getting nqueue csvs for LIFO\n")
    df_nqueue_mean_lifo.to_csv('df_nqueue_mean_lifo.csv', sep='\t', encoding='utf-8')
    df_nqueue_var_chi_lifo.to_csv('df_nqueue_var_chi_lifo.csv', sep='\t', encoding='utf-8')
    df_nqueue_var_t_lifo.to_csv('df_nqueue_var_t_lifo.csv', sep='\t', encoding='utf-8')


"""#### Código de retorno de resultados

"""

def get_results_all_table(seed, k_min, queue_discipline=FIFO, print_graph=False):
  # Seed do gerador de numeros aleatorios
  random.seed(seed)
  # Lista de taxas de utilizacao possives
  exp_functions = [exp02, exp04, exp06, exp08, exp09]
  results_utilization_all = []
  # Para cada indice e cada taxa de utilizacao
  for i, exp_function in enumerate(exp_functions):
    # Pega o intevalo de confianca da media do tempo de espera
    results_all = simulate(NSAMPLES, k_min, queue_discipline, exp_function, exp10, graph_values=print_graph)
    results_utilization_all.append(results_all)
  return results_utilization_all

"""#### Código de geração de data_frame_dict"""

def get_all_tables_utilization_all(results_by_utilizations, queue_discipline):
  options = {"nqueue_var_chi","nqueue_mean", "time_var_chi", "time_mean"}
  df_dict = {
      "nqueue_var_chi": pd.DataFrame(columns=["Utilization", "Discipline", "Lower Limit", "Upper Limit", "Precision", "Center", "Analytic Value"]),
      "nqueue_mean": pd.DataFrame(columns=["Utilization", "Discipline", "Lower Limit", "Upper Limit", "Precision", "Center", "Analytic Value"]), 
      "time_var_chi": pd.DataFrame(columns=["Utilization", "Discipline", "Lower Limit", "Upper Limit", "Precision", "Center", "Analytic Value"]),
      "time_mean": pd.DataFrame(columns=["Utilization", "Discipline", "Lower Limit", "Upper Limit", "Precision", "Center", "Analytic Value"])
  }
  utilization_labels = ["0.2", "0.4", "0.6", "0.8", "0.9"]
  for i, utilization_label in enumerate(utilization_labels):
      print(i, utilization_label)
      for option in options:
          results = deque(results_by_utilizations[i][option])
          results.appendleft(utilization_label)
          if option == 'nqueue_var_chi':
              results.append(an_nq_var[i])
          elif option == 'nqueue_mean':
              results.append(an_nq_mean[i])
          elif option == 'time_var_chi':
              if queue_discipline == FIFO:
                  results.append(an_w_var_fcfs[i])
              else:
                  results.append(an_w_var_lcfs[i])
          elif option == 'time_mean':
              results.append(an_w_mean[i])
          df_dict[option].loc[len(df_dict[option].index)] = results
  return df_dict