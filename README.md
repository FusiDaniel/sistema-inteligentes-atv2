# Instruções de uso
Instale as dependências presentes no arquivo requirements.txt.

Caso queira treinar uma rede do zero, siga todos os próximos passos a seguir.

Caso queira pular a primeira etapa do treino, copie para a raiz o arquivo neat-checkpoint-1959 presente na pasta checkpoint.

Caso queira pular a segunda etapa do treino, copie para a raiz o arquivo neat-checkpoint-2149 presente na pasta checkpoint.

Caso queira pular todo o treinamento e salvamento da rede, copie para a raiz o arquivo best.pickle presente na pasta checkpoint

## Primeira etapa de treinamento
Dentro de main.py, troque train_step para 1, rode o arquivo, e selecione "Selecionar melhor rede usando NEAT", e deixe em branco o checkpoint.

Deixe rodando até que o fitness de alguma espécie chegue em 47 (fitness máximo para essa etapa).

Depois de chegar no fitness de 47, pare de rodar o programa e anote o checkpoint em que ele chegou nesse fitness.

## Segunda etapa de treinamento
Dentro de main.py, troque train_step para 2, rode o arquivo, e selecione "Selecionar melhor rede usando NEAT", e coloque apenas o número do checkpoint gerado na última etapa.

Deixe rodando até que o fitness de alguma espécie chegue em torno de 200 (fitness máximo para essa etapa).

Depois de chegar num fitness em torno de 200, pare de rodar o programa e anote o checkpoint em que ele chegou nesse fitness.

## Salvar rede
Rode o arquivo main.py e selecione "Salvar melhor rede a partir de checkpoint". Digite o número do checkpoint adquirido na etapa anterior. Ele gerará um arquivo chamado best.pickle na raiz do projeto

## Rodar aplicação no terminal
Certifique-se que existe um arquivo best.pickle na raiz do projeto, rode o arquivo main.py e selecione "Testar rede salva em best.pickle".

## Rodar aplicação no Envisim
Certifique-se que existe um arquivo best.pickle na raiz do projeto, rode o Agent_Client_main.py. 

## Geração de outros mapas aleatórios
Caso queira gerar outros mapas aleatórios para a segunda etapa de treino, rode o arquivo main.py, selecione "Gerar mapa novo" e siga as instruções do terminal.

