library(ggplot2)
library(wordcloud)
library(wordcloud2)
library(tm)
library(RColorBrewer)
library(tidyverse)

df <- read.csv('/mnt/3602F83B02F80223/Downloads/imperial/Sem 2 - Machine Learning/Project/Data/lemma_dfUadm.csv')

ggplot(df, aes(x=DAYS_NEXT_UADM)) +
    geom_histogram() +
    scale_x_log10()

ggplot(df, aes(x=DAYS_NEXT_UADM)) +
    geom_histogram(binwidth = 30) +
    xlim(3365, 4500)


#### wordcloud viz ####
cases <- df %>%
    filter(TARGET == 'True')

controls <- df %>%
    filter(TARGET == 'False')

docs_full <- Corpus(VectorSource(df$TEXT_CONCAT))
dtm_full <- TermDocumentMatrix(docs_full)
sparse_full <- removeSparseTerms(dtm_full, 0.95)

matrix_full <- as.matrix(sparse_full) 
words_full <- sort(rowSums(matrix_full),decreasing=TRUE) 
df_full <- data.frame(word = names(words_full),freq=words_full)

# cases wordcloud
docs <- Corpus(VectorSource(cases$TEXT_CONCAT))

dtm <- TermDocumentMatrix(docs)
matrix <- as.matrix(dtm) 
words <- sort(rowSums(matrix),decreasing=TRUE) 
df_cases <- data.frame(word = names(words),freq=words)

# drop less than 5 freq for viz 
df_cases_top <- df_cases %>%
    filter(freq >= 5)

df_cases_top2 <- df_cases %>%
    filter(freq > 1000)

wordcloud(words = df_cases_top$word, 
     freq = df_cases$freq, min.freq = 3,
     max.words=2000,
     random.order=TRUE,
     rot.per=0.35,
     colors=brewer.pal(8, "Oranges")
)

cross_fig = '/mnt/3602F83B02F80223/Downloads/imperial/Sem 2 - Machine Learning/Project/Data/wordcloud_mask/medical-cross-symbol.png'
wordcloud2(data=df_cases, size=1, figPath = cross_fig, color='random-dark')

wordcloud2(data=df_cases_top2, size=1, color='orange', backgroundColor='#EEECE2') 

    #gridSize=c(500,300))

wordcloud2(demoFreq, figPath = cross_fig, size = 1.5, color = "skyblue", backgroundColor="black")
wordcloud2(demoFreq, size = 0.7, shape = 'star')

wordcloud2(data=df, size=1.6, color='random-light', backgroundColor = 'black')


wordcloud2(data=df, size=1.6, color='random-dark')


# cases wordcloud
docs_controls <- Corpus(VectorSource(controls$TEXT_CONCAT))

dtm_controls <- TermDocumentMatrix(docs_controls)
sparse_controls <- removeSparseTerms(dtm_controls, 0.95) # save memory 

matrix_controls <- as.matrix(sparse_controls) 
words_controls <- sort(rowSums(matrix_controls),decreasing=TRUE) 
df_controls <- data.frame(word = names(words_controls),freq=words_controls)

# drop less than 5 freq for viz 
df_controls_top <- df_controls %>%
    filter(freq > 1000)


wordcloud2(data=df_controls_top, size=1, color='skyblue', backgroundColor='#EEECE2') 
