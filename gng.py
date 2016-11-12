from random import randint
import genre
import cluster

genre_list = ['blues','rock','edm','jazz','classical']
train_data = []
train_target = []
for genre_name in genre_list:
	genre_class = genre.genre(genre_name,400)
        temp = genre_class.get_train()
	train_data = train_data + temp[0]
	train_target = train_target + temp[1]
       
clusters = []
index1 = randint(0,len(train_data)/5)
index2 = randint(len(train_data)/2,len(train_data)-1)

clusters.append(cluster.cluster(train_data[index1], train_target[index1]))
clusters.append(cluster.cluster(train_data[index2], train_target[index2]))

for i in range(0,len(train_data)):
    val = 0;
    min_distance = 10000
    cluster_to_be_added = 0
    for cur_cluster in clusters:
	dist = cur_cluster.try_fit(train_data[i], train_target[i])
	if(dist==-1):
		continue
        elif(dist<min_distance):
	        min_distance = dist
		cluster_to_be_added = cur_cluster
		val = 1
    if(val==0):
	clusters.append(cluster.cluster(train_data[i],train_target[i]))
    else:
	cluster_to_be_added.add(train_data[i], train_target[i])

for cur_cluster in clusters:
    print(cur_cluster.vectors_labels)

print(len(clusters))
