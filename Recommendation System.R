#load library
library('recommenderlab')
library('ggplot2')
library('data.table')
library('reshape2')

#load data
movie_data <- read.csv('movies.csv',stringsAsFactors = FALSE)
rating_data <- read.csv('ratings.csv')

#data pre-processing (one hot encoding)
movie_genre <- as.data.frame(movie_data$genres, stringsAsFactors = FALSE)

movie_genre2 <- as.data.frame(tstrsplit(movie_genre[,1], '[|]', type.convert = TRUE),
                              stringsAsFactors = FALSE)

colnames(movie_genre2) <- c(1:10)

list_genre <- c("Action","Adventure","Animation","Children",
                "Comedy","Crime","Documentary","Drama","Fantasy",
                "Film-Noir","Horror","Musical","Mystery","Romance",
                "Sci-Fi","Thriller","War","Western")

genre_mat1 <- matrix(0,10330,18)
genre_mat1[1,] <- list_genre
colnames(genre_mat1) <- list_genre

for (index in 1:nrow(movie_genre2)) {
  for (col in 1:ncol(movie_genre2)) {
    gen_col = which(genre_mat1[1,] == movie_genre2[index,col])
    genre_mat1[index+1, gen_col] <- 1
  }
}
genre_mat2 <-  as.data.frame(genre_mat1[-1,], stringsAsFactors = FALSE)
for (col in 1:ncol(genre_mat2)) {
  genre_mat2[,col] <- as.integer(genre_mat2[,col])
}

#create a search matrix to easily search any film by specifying the genre
SearchMatrix <- cbind(movie_data[,1:2], genre_mat2[])

#create sparse matrix to cater for movies with several genres (matrix with few non-zero values)
ratingMatrix <- dcast(rating_data, userId~movieId, value.var = "rating", na.rm = FALSE)
ratingMatrix <- as.matrix(ratingMatrix[,-1])
#Convert rating matrix into a recommenderlab sparse matrix
ratingMatrix <- as(ratingMatrix, "realRatingMatrix")


#Let us now overview some of the important parameters that provide us various options for building recommendation systems for movies
recommendation_model <- recommenderRegistry$get_entries(dataType = "realRatingMatrix")
names(recommendation_model)

lapply(recommendation_model, "[[", "description")

#using Item Based Collaborative Filtering
recommendation_model$IBCF_realRatingMatrix$parameters

#Exploring similar data (users)
similarity_mat <- similarity(ratingMatrix[1:4,],
                             method = "cosine",
                             which = "users")
as.matrix(similarity_mat)
par(mar = rep(2, 4))
image(as.matrix(similarity_mat), main = "User's Similarities")

#Exploring similar data (movies)
movie_similarity <- similarity(ratingMatrix[1:4,],
                             method = "cosine",
                             which = "items")
as.matrix(movie_similarity)
par(mar = rep(2, 4))
image(as.matrix(movie_similarity), main = "Movies Similarities")

#extract most unique ratings
rating_values <- as.vector(ratingMatrix@data)
unique(rating_values)

#create a table of ratings that display the most unique ratings
Table_of_Ratings <- table(rating_values)

#most viewed movies 
movie_views <- colCounts(ratingMatrix) #count views for each movie

table_views <- data.frame(movie = names(movie_views), views = movie_views)
table_views <- table_views[order(table_views$views,
                                 decreasing = TRUE), ] # sort by number of views
table_views$title <- NA
for (index in 1:10325){
  table_views[index,3] <- as.character(subset(movie_data,
                                              movie_data$movieId == table_views[index,1])$title)
}
table_views[1:6,]

#visualize top films
ggplot(table_views[1:6, ], aes(x = title, y = views)) +
  geom_bar(stat = "identity", fill = 'steelblue') +
  geom_text(aes(label=views), vjust=-0.3, size=3.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle("Total Views of the Top Films")

#HeatMap of Movie Ratings
image(ratingMatrix[1:20, 1:25], axes = FALSE, main = "Heatmap of the first 25 rows and 25 columns")

#Data Preparation

#set threshold for the minimum number of users who have rated
#a film as 50. This is also same for minimum number of views that are per film.
#This way, we have filtered a list of watched films from least-watched ones. 

movie_ratings <- ratingMatrix[rowCounts(ratingMatrix) > 50,
                              colCounts(ratingMatrix) > 50]

#delineate matrix of relevant users
minimum_movies<- quantile(rowCounts(movie_ratings), 0.98)
minimum_users <- quantile(colCounts(movie_ratings), 0.98)
image(movie_ratings[rowCounts(movie_ratings) > minimum_movies,
                    colCounts(movie_ratings) > minimum_users],
      main = "Heatmap of the top users and movies")

#visualize the average ratings per user
average_rating <- rowMeans(movie_ratings)
qplot(average_rating, fill = I ("steelblue"), col = I("red")) +
  ggtitle("Distribution of the average rating per user")

#Normalization
normalized_ratings <- normalize(movie_ratings)
sum(rowMeans(normalized_ratings) > 0.00001)

image(normalized_ratings[rowCounts(normalized_ratings) > minimum_movies,
                         colCounts(normalized_ratings) > minimum_users],
      main = "Normalized Ratings of the Top Users")

#perform data binarization (two discrete values, 0 & 1)
binary_minimum_movies <- quantile(rowCounts(movie_ratings), 0.95)
binary_minimum_users <- quantile(colCounts(movie_ratings), 0.95)
#movies_watched <- binarize(movie_ratings, minRating = 1)

good_rated_films <- binarize(movie_ratings, minRating = 3)
image(good_rated_films[rowCounts(movie_ratings) > binary_minimum_movies,
                       colCounts(movie_ratings) > binary_minimum_users],
      main = "Heatmap of the top users and movies")

#Collaborative Filtering System
sampled_data <- sample(x = c(TRUE, FALSE),
                      size = nrow(movie_ratings),
                      replace = TRUE,
                      prob = c(0.8,0.2))
training_data <- movie_ratings[sampled_data, ]
testing_data <- movie_ratings[!sampled_data, ]

#build recommendation system
#k denotes the number of items for computing their similarities
recommendation_system <- recommenderRegistry$get_entries(dataType = "realRatingMatrix")
recommendation_system$IBCF_realRatingMatrix$parameters

recommen_model <- Recommender(data = training_data,
                              method = "IBCF",
                              parameter = list(k=30))
recommen_model
class(recommen_model)

#use getModel() to retrieve the model
model_info <- getModel(recommen_model)
class(model_info$sim) #contains similarity matrix
dim(model_info$sim)
top_items <- 20
image(model_info$sim[1:top_items, 1:top_items],
      main = "Heatmap of the first rows and columns")

sum_rows <- rowSums(model_info$sim > 0)
table(sum_rows)

sum_cols <- colSums(model_info$sim > 0)
qplot(sum_cols, fill =I("steelblue"), col =I("red")) +
  ggtitle("Distribution of the column count")

#recommendation system
top_recommendations <- 10 #number of items to recommend to each user
predicted_recommendations <- predict(object = recommen_model,
                                     newdata = testing_data,
                                     n = top_recommendations)
predicted_recommendations

user1 <- predicted_recommendations@items[[1]] #recommendation for the first user
movies_user1 <- predicted_recommendations@itemLabels[user1]
movies_user2 <- movies_user1
for (index in 1:10){
  movies_user2[index] <- as.character(subset(movie_data,
                                             movie_data$movieId == movies_user1[index])$title)
}
movies_user2

#matrix with the recommendations for each user
recommendation_matrix <- sapply(predicted_recommendations@items,
                                function(x){
                                  as.integer(colnames(movie_ratings)[x])
                                })
recommendation_matrix[,1:4]
number_of_items <- factor(table(recommendation_matrix))
qplot(number_of_items, fill=I("steelblue"), col=I("red")) + ggtitle("Distribution of the Number of Items for IBCF")


number_of_items_sorted <- sort(number_of_items, decreasing = TRUE)
number_of_items_top <- head(number_of_items_sorted, n = 4)
table_top <- data.frame(as.integer(names(number_of_items_top)),
                        number_of_items_top)
for(i in 1:4) {
  table_top[i,1] <- as.character(subset(movie_data,
                                        movie_data$movieId == table_top[i,1])$title)
}

colnames(table_top) <- c("Movie Title", "No. of Items")
head(table_top)
