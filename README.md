# 🐱🐶🦜 Transfer Learning — Classificador de Animais

> Projeto desenvolvido como parte do aprendizado em **Deep Learning** na disciplina de Machine Learning.  
> Utiliza Transfer Learning com a rede VGG16 para classificar imagens de gatos, cachorros e pássaros.

---

## 📌 Sobre o projeto

Este repositório documenta minha jornada de aprendizado em **Transfer Learning**, uma técnica de Deep Learning que reaproveita o conhecimento de redes neurais já treinadas para resolver novos problemas com menos dados e menos tempo de treinamento.

O projeto foi desenvolvido em Python no ambiente **Google Colab**, com base em um código fornecido pelo professor e expandido com novos datasets, técnicas de Data Augmentation e Fine-Tuning.

### O que o modelo faz?

Dado uma imagem de entrada, o modelo identifica se ela contém um dos 7 animais abaixo:

| Classe | Animal | Dataset de origem |
|--------|--------|-------------------|
| `cat` | 🐱 Gato | Animals-10 (Kaggle) |
| `dog` | 🐶 Cachorro | Animals-10 (Kaggle) |
| `painted_bunting` | 🦜 Pintassilgo | CUB-200-2011 (Caltech) |
| `kingfisher` | 🐦 Martim-pescador | CUB-200-2011 (Caltech) |
| `horned_lark` | 🐦 Cotovia | CUB-200-2011 (Caltech) |
| `house_sparrow` | 🐦 Pardal | CUB-200-2011 (Caltech) |
| `yellowthroat` | 🐦 Maritaca-amarela | CUB-200-2011 (Caltech) |

---

## 🧠 O que é Transfer Learning?

Imagine que você já sabe jogar squash e decide aprender tênis. Você não precisa aprender do zero o que é uma raquete, como se movimentar na quadra ou como ter foco — esse conhecimento **transfere**.

Transfer Learning faz o mesmo com redes neurais:

```
ImageNet (1,2 milhão de imagens)
        ↓
    VGG16 aprende a reconhecer
    bordas, texturas, formas, objetos
        ↓
    Aproveitamos esse conhecimento
        ↓
    Adicionamos uma camada nova
    para nossas 7 classes
        ↓
    Treinamos só essa última parte
    com nossos ~4.200 imagens
```

**Por que isso é poderoso?** Sem Transfer Learning, precisaríamos de centenas de milhares de imagens e dias de treinamento. Com ele, chegamos a ~80% de acurácia com ~600 imagens por classe em poucas horas.

---

## 🏗️ Arquitetura do modelo

```
Entrada: imagem 224×224×3 (RGB)
    │
    ▼
┌─────────────────────────────────────┐
│  VGG16 — 13 camadas convolucionais  │  ← CONGELADA (pesos do ImageNet)
│  + 2 camadas densas internas        │    138 milhões de parâmetros
└─────────────────────────────────────┘
    │
    ▼
GlobalAveragePooling2D   ← Reduz (7×7×512) → vetor de 512 valores
    │
    ▼
Dense(512, relu)         ← Aprende padrões específicos dos nossos animais
    │
    ▼
Dropout(0.5)             ← Evita overfitting (desliga 50% dos neurônios no treino)
    │
    ▼
Dense(7, softmax)        ← Saída: probabilidade para cada uma das 7 classes
```

**Parâmetros treináveis:** ~397.000 (apenas a última camada)  
**Parâmetros congelados:** ~14.700.000 (toda a VGG16)

---

## 🔄 Fases de treinamento

O treinamento foi dividido em duas fases:

### Fase 1 — Feature Extraction (15 épocas)
A VGG16 fica completamente congelada. Apenas as camadas novas que adicionamos são treinadas. É como usar a VGG16 como um "extrator de características" fixo.

### Fase 2 — Fine-Tuning (10 épocas)
Descongelamos o **bloco 5** da VGG16 (as últimas 4 camadas convolucionais) e deixamos que elas se ajustem levemente ao nosso dataset, com uma taxa de aprendizado 10× menor para não perder o conhecimento já adquirido.

---

## 📊 Resultados

| Modelo | Acurácia no teste |
|--------|-------------------|
| Rede treinada do zero (baseline) | ~49% |
| VGG16 + Transfer Learning | ~80% |

A melhoria de **~31 pontos percentuais** com menos parâmetros treináveis demonstra o poder do Transfer Learning, especialmente quando se tem poucos dados.

---

## 🗂️ Estrutura do repositório

```
📦 transfer-learning-animais/
 ┣ 📓 transfer_learning_gatos_cachorros_passaros.ipynb  ← Notebook principal
 ┣ 📄 README.md                                         ← Este arquivo
 ┣ 📄 RELATORIO.md                                      ← Relatório de aprendizado
 ┗ 📁 imagens/
    ┣ 🖼️ 1.png                         ← Imagem geral de treinamento
    ┗ 🖼️ 2.png                         ← Imagem de teste de pássaro
    ┗ 🖼️ 3.png                         ← Curvas de aprendizado
    ┗ 🖼️ 4.png                         ← Matriz de Confusão
    ┗ 🖼️ 5.png                         ← Imagem de teste de previsão
    ┗ 🖼️ 6.png                         ← Imagem de teste de pássaro
    ┗ 🖼️ 7.png                         ← Imagem de teste de pássaro
    ┗ 🖼️ 8.png                         ← Imagem de teste de pássaro
    ┗ 🖼️ 9.png                         ← Imagem de teste de pássaro
    ┗ 🖼️ 10.png                        ← Imagem de teste de pássaro
    ┗ 🖼️ 11.png                        ← Imagem de teste de pássaro
    ┗ 🖼️ 12.png                        ← Imagem de teste de pássaro
    ┗ 🖼️ 13.png                        ← Imagem de teste de pássaro
    ┗ 🖼️ 14.png                        ← Imagem de teste de pássaro
    ┗ 🖼️ 15.png                        ← Imagem de teste de pássaro

```

> **Nota:** As imagens do dataset não estão incluídas no repositório (são grandes demais). O notebook baixa tudo automaticamente ao ser executado.

---

## 🚀 Como executar

### Pré-requisitos
- Conta Google (para o Colab)
- Conta Kaggle (gratuita, para baixar o Animals-10)

### Passo a passo

**1.** Clique no botão abaixo para abrir o notebook direto no Colab:

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SEU_USUARIO/transfer-learning-animais/blob/main/transfer_learning_gatos_cachorros_passaros.ipynb)

**2.** Ative a GPU: `Ambiente de execução → Alterar tipo de ambiente de execução → GPU T4`

**3.** Execute as células em ordem (de cima para baixo)

**4.** Na Célula 4, quando solicitado, autentique sua conta Kaggle

> 💡 O tempo total de execução é de aproximadamente 1–2 horas com GPU T4 gratuita do Colab.

---

## 📦 Datasets utilizados

| Dataset | Fonte | Licença | Uso |
|---------|-------|---------|-----|
| Animals-10 | [Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10) | CC BY-SA 4.0 | Gatos e cachorros |
| CUB-200-2011 | [Caltech](https://www.vision.caltech.edu/datasets/cub_200_2011/) | Acadêmico | 5 espécies de pássaros |

---

## 🛠️ Tecnologias

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-VGG16-red?logo=keras)
![Colab](https://img.shields.io/badge/Google_Colab-GPU_T4-yellow?logo=googlecolab)

- **Python 3.10**
- **TensorFlow / Keras** — framework de Deep Learning
- **VGG16** — rede pré-treinada no ImageNet
- **Google Colab** — ambiente de execução com GPU gratuita
- **Matplotlib / Seaborn** — visualizações
- **scikit-learn** — métricas de avaliação

---

## 👨‍🎓 Contexto acadêmico

Este projeto foi desenvolvido como parte de uma colaboração **Open Source** proposta pelo professor da disciplina de Machine Learning. O objetivo é documentar o processo de aprendizado de forma transparente e compartilhada, permitindo que colegas reproduzam e contribuam com os experimentos.

---

## 📚 Referências

- SIMONYAN, K.; ZISSERMAN, A. **Very Deep Convolutional Networks for Large-Scale Image Recognition**. ICLR, 2015.
- TORREY, L.; SHAVLIK, J. **Transfer Learning**. Handbook of Research on Machine Learning Applications, 2010.
- [Documentação oficial do Keras — Transfer Learning](https://keras.io/guides/transfer_learning/)
- [CS231n Stanford — Convolutional Neural Networks](http://cs231n.stanford.edu/)

---

*Projeto desenvolvido com fins educacionais — fique à vontade para abrir issues, sugerir melhorias ou fazer um fork!* 🤝
