# Commands Categorization

Este documento está disponível em [Português](https://github.com/lucsalm/commands-categirization/blob/main/README-pt-BR.md), porém também
está disponível em [Inglês](https://github.com/lucsalm/commands-categirization/blob/main/README.md).

## Visão Geral

Este projeto é base do meu
[Trabalho de Conclusão de Curso](https://github.com/lucsalm/commands-categirization/blob/main/TCC.pdf)
da minha graduação [Bacharelado em Matemática Aplicada e Computacional](https://www.ime.usp.br/bmac/)
no [Instituto de Matemática e Estatística](https://www.ime.usp.br) da [USP](https://www5.usp.br).
O trabalho é focado no estudo da arquitetura Transformer e sua aplicação na categorização de comandos de fala
que por sua vez é um processo que envolve identificar e classificar os diferentes tipos de instruções ou solicitações
emitidas por meio da fala humana. O repositório contém as implementações utilizadas,
desde o pré-processamento dos dados, criação do modelo até as definições de treinamento e
geração de gráficos para avaliação de resultados.

## Tecnologias

![Python](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white)
![Tensorflow](https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Tensorflow](https://img.shields.io/badge/Keras-D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Numpy](https://img.shields.io/badge/NumPy-013243.svg?style=for-the-badge&logo=NumPy&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000.svg?style=for-the-badge&logo=Flask&logoColor=white)
![Latex](https://img.shields.io/badge/LaTeX-008080.svg?style=for-the-badge&logo=LaTeX&logoColor=white)

## Dados

Por conta da natureza do problema estudado, os dados seguem o padrão de pares de:

- ### Áudios
  Os áudios seguem a representação unidimensional em formato de onda extraídos de arquivos WAV:
  ![WAV](https://raw.githubusercontent.com/lucsalm/commands-categirization/main/app/files/documentation/wav_all.png)
  Porém na fase de pré-processamento eles são convertidos para o formato bidimensional de espectrograma STFT (Short-time
  Fourier transform):
  ![STFT](https://raw.githubusercontent.com/lucsalm/commands-categirization/main/app/files/documentation/spec_all.png)

- ### Rótulos
  São a representação numérica para os comandos, os indicies definidos são os seguintes:

  | **Indice** | 1    | 2  | 3    | 4  | 5     | 6    | 7  | 8   |
  |------------|------|----|------|----|-------|------|----|-----|
  | **Texto**  | Down | Go | Left | No | Right | Stop | Up | Yes |

  A partir dos indices definidos, os comandos são identificados por um vetor cuja os valores somam um, ao representar
  um comando, seu indice será o maior valor do vetor, por exemplo, uma representação do comando *"No"* pode ser feita
  por:

  | **Indice** | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
  |------------|---|---|---|---|---|---|---|---|
  | **Vetor**  | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |

Os pares são divididos com as mesmas proporções para cada comando em três conjuntos:

- ### Conjuntos
  | **Nome**    | Treinamento | Validação  | Teste     |
  |-------------|-------------|------------|-----------|
  | **Tamanho** | 6400 (80%)  | 800 (10%)  | 800 (10%) |

## Modelo

- ### Arquitetura
  A arquitetura do modelo utilizada foi baseada na arquitetura Transformer, em especial no conceito do Encoder
  e no uso do mecanismo de Self-Attention, a representação em forma de fluxograma pode ser visualizada pelo desenho:
  ![Model](https://raw.githubusercontent.com/lucsalm/commands-categirization/main/app/files/documentation/model-diagram.jpg)

  Cada camada presente na arquitetura tem um papel específico para o modelo, esses estão devidamente documentados no
  trabalho, para aprofundar os
  detalhes consulte a sessão **(3.1) Camadas**.

## Treinamento

- ### Definições

    - **Função de Perda**: Categorical Crossentropy
    - **Função de Precisão**: Categorical Accuracy
    - **Otimizador**: Adaptative Moment Estimation
    - **Hiperparêmetros**:
        - Epócas: 400
        - Batch size: 32
        - Heads: 2
        - Dropout rate: 0.1

  Cada definição escolhida tem um funcionamento específico, esses estão devidamente documentados no trabalho, para
  aprofundar os detalhes
  consulte a sessão **(3.2) Treinamento**.


- ### Escolha
  O critério de escolha dos pesos que definem o modelo é feito a partir da época onde for encontrada a maior precisão
  para o conjunto de validação.

## Resultados

- ### Precisão
  | Conjunto     | Treinamento | Validação | Teste  |
  |--------------|-------------|-----------|--------|
  | **Precisão** | 99.08%      | 85.25%    | 85.87% | 

- ### Comportamento
  - **Função de Perda** (Treinamento) 
  
      ![Perda](https://raw.githubusercontent.com/lucsalm/commands-categirization/main/app/files/documentation/train_loss.png)

  - **Função de Precisão** (Validação)
  
      ![Precisao](https://raw.githubusercontent.com/lucsalm/commands-categirization/main/app/files/documentation/validation_accuracy.png)
  
  - **Matriz de confusão** (Teste)
  
      ![Confusao](https://raw.githubusercontent.com/lucsalm/commands-categirization/main/app/files/documentation/confusion-teste.png)

## Captura de Tela de Exemplo

![Screenshot](https://raw.githubusercontent.com/lucsalm/commands-categirization/main/app/files/documentation/screenshot.png)

## Como usar?
1. Certifique-se de que o Docker esteja instalado em sua máquina.
2. Clone este repositório para o seu ambiente local.
3. Navegue até o diretório do projeto.
4. No terminal, execute o seguinte comando para construir e iniciar o contêiner Docker:
    - No Linux, execute:
        ```bash
        docker compose up
        ```

    - No Windows, execute:
        ```bash
        docker-compose up
        ```

5. Após a construção do contêiner e a inicialização da aplicação, acesse `http://localhost:5000` em seu navegador da web para explorar.

**Observação:** 
- Certifique-se de que a porta `5000` não está sendo utilizada por outra aplicação em seu sistema para evitar conflitos. Se necessário, você pode modificar o mapeamento de porta no arquivo `docker-compose.yml`.


## Referências
- Shafran, I., Riley, M., & Mohri, M. (2003). Voice signatures. In: 2003 IEEE workshop on automatic speech recognition and understanding (IEEE Cat. No. 03EX721), pp. 31–36.


- Sivanandam, S., & Paulraj, M. (2009). Introduction to artificial neural networks. Vikas Publishing House.


- Vaswani, A., et al. (2017). Attention is all you need. Advances in neural information processing systems 30.

