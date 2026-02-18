# --- Install Libraries --- #
install.packages(c('tidyverse', 'tidytext', 'textstem', 'tokenizers', 
                   'reshape2', 'wordcloud'))

# --- Libraries --- #
library(tidyverse)  # Handle data
library(tidytext)   # work with text - main functionality
library(textstem)   # stem words
library(tokenizers) # count words
library(reshape2)   # cast from long to wide and vice versa
library(wordcloud)  # plot wordclouds

# --- Load Data --- #
disney_df <- read.csv('../data/DisneylandReviews.csv', encoding = 'latin1')

# --- Assign pos/neg/neutral to review based on rating --- #
# 1 or 2 = negative / 3 = neutral / 4 or 5 = positive

disney_df <- disney_df %>% 
  mutate(sentiment = case_when(
    Rating > 3 ~ 'Positive',
    Rating == 3 ~ 'Neutral',
    Rating < 3 ~ 'Negative'
  ))

# --- Counting Characters and Approx. Word Counts --- #
# Count Words and characters
disney_df <- disney_df %>% 
  mutate(n_words = count_words(Review_Text), # create column with word count
         n_char = nchar(Review_Text))        # create column with character count

  
# --- Reviews to tokens --- #
  
tokens <- disney_df %>% 
  unnest_tokens(word, Review_Text)

common_words <- tokens %>% # Create dataset t
  group_by(word) %>% 
  count(sort = TRUE)

# --- Standard Stop Words --- # 

tokens_no_stop <- tokens %>% 
  anti_join(stop_words) %>% 
  filter(!str_detect(word, "[[:digit:]]+"))

common_words_2 <- tokens_no_stop %>% 
  group_by(word) %>% 
  count(sort = TRUE)

#--- Custom Stop Words --- #
custom_stop_words <-  # words that have no real value in Disney reviews
  tibble(
    word = c(
      'park',
      'parks',
      'disney',
      'disneyland',
      'paris',
      'california',
      'florida',
      'world',
      'land',
      'visit',
      'visits',
      'visited',
      'trip',
      'day',
      'days',
      'time',
      'times',
      'people',
      'lot',
      'lots',
      'bit',
      'main',
      'start',
      'book',
      'absolutely',
      'taking',
      'managed',
      'decided',
      'makes',
      'real',
      'life',
      'ago',
      # also remove sentiment fillers
      'amazing',
      'love',
      'loved',
      'magival',
      'magic',
      'nice',
      'enjoyed',
      'enjoy',
      'worth',
      'recommend',
      'beautiful',
      'happiest',
      'excellent',
      'favorit',
      'lovely',
      'perfect',
      'highly'
      
    ),
    lexicon = 'docs' # Create lexicon !!!!
  )

# remove custom stopwords

tokens_no_stop <- tokens_no_stop %>%
  anti_join(custom_stop_words)

# Plot Common Words after stopword removal
tokens_no_stop %>%
  group_by(word) %>% 
  count(sort = TRUE) %>% 
  ungroup() %>% 
  top_n(25) %>%
  ggplot(aes(x = n, 
             y = reorder(word, n)
  )
  ) +
  geom_col() +
  scale_y_reordered() +
  labs(y = NULL)

# --- Stemming --- #

tokens_stemmed <- tokens_no_stop %>% 
  mutate(word_stem = stem_words(word),
         word_lemma = lemmatize_words(word)) 


# --- Word Clouds --- #

tokens_stemmed %>%
  filter(sentiment != 'Neutral') %>% 
  count(word_lemma, sentiment, sort = TRUE) %>%
  acast(word_lemma ~ sentiment, value.var = "n", fill = 0) %>%
  commonality.cloud(max.words = 75)

# --- Establish a Vocab List --- #

word_counts <- tokens_stemmed %>% 
  group_by(word_lemma) %>% 
  count(sort = TRUE) %>% 
  filter(n >= 10) # reduce noise / model will be faster

tokens_filtered <- tokens_stemmed %>% 
  filter(word_lemma %in% word_counts$word_lemma)

# --- Put the reviews back together --- #

reviews_processed <- tokens_filtered %>% 
  group_by(Review_ID) %>% 
  summarise(
    review_clean = str_c(word_lemma, collapse = ' '),
    .groups = 'drop'
  )

reviews_processed <- disney_df %>% 
  left_join(reviews_processed, by ='Review_ID')

# --- Save Data --- #
write_csv(reviews_processed, '../data/DisneyReviews_processed.csv', na = "")

