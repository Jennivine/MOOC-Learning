import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#find the distance betwee  two points
def distance(p1,p2):
    '''Find the distance between points p1 and p2'''
    return np.sqrt(np.sum(np.power(p2-p1, 2)))

#KNN CLASSFIER
#finds the most common element in votes
def majority_vote(votes):
    '''return the most common element in votes'''
    vote_counts = {}
    
    for vote in votes:
        if vote in vote_counts:
            vote_counts[vote] += 1
        else:
            vote_counts[vote] = 1
            
    winners = []
    max_result = max(vote_counts.values())
    for vote, count in vote_counts.items():
        if count == max_result:
            winners.append(vote)
            
    return random.choice(winners)

#shorter and faster version of the code above, however it
#always return the smallest value if a tie occurs given the
#nature of "Mode"
import scipy.stats as ss
def majority_vote_short(votes):
    '''return the most common element in votes'''
    mode, count = ss.mstats.mode(votes)           
    return mode

#Find the nearest neighbours of this newly plotted point
def find_nearest_neighbours(p, points, k=5):
    '''find the k nearest neighbours of point p and return their indices'''
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p,points[i])
    ind = np.argsort(distances)
    return ind[:k]
    
#Then sort this point into the classes based on the classes of most point nearest to it
def knn_predict(p, points, outcomes, k=5):
    ind = find_nearest_neighbours(p, points, k)
    return majority_vote(outcomes[ind])

#Artificial data generated with help of computer
def generate_synth_data(n=50):
    '''Create two sets of points from bivariate normal distribution'''
    points = np.concatenate((ss.norm(0,1).rvs((n,2)), ss.norm(1,1).rvs((n,2))), axis = 0)
    outcomes = np.concatenate((np.repeat(0, n), np.repeat(1, n)))
    return (points, outcomes)

#Plot prediction grid
def make_prediction_grid(predictors, outcomes, limits, h, k):
    '''classify each point on the prediction grid'''
    (x_min, x_max, y_min, y_max) = limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx,yy = np.meshgrid(xs,ys)

    prediction_grid = np.zeros(xx.shape, dtype = int)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p = np.array([x,y])
            prediction_grid[j,i] = knn_predict(p, predictors, outcomes, k)

    return (xx, yy, prediction_grid)
            
def plot_prediction_grid (xx, yy, prediction_grid, filename):
    """ Plot KNN predictions for every point on the grid."""
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    plt.savefig(filename)

#uses our KNN_predictor to classify different iris flowers   
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
iris = datasets.load_iris()

#plot all iris flower data on a grid
predictors = iris.data[:, 0:2]
outcomes = iris.target
plt.plot(predictors[outcomes == 0][:,0], predictors[outcomes == 0][:,1],"ro")
plt.plot(predictors[outcomes == 1][:,0], predictors[outcomes == 1][:,1],"go")
plt.plot(predictors[outcomes == 2][:,0], predictors[outcomes == 2][:,1],"bo")
plt.savefig("iris.pdf")

#determines the types of iris it belongs to
k = 5; filename = "iris_grid.pdf"; limits = (4,8,1.5,4.5); h = 0.1
(xx,yy,prediction_grid) = make_prediction_grid(predictors, outcomes,limits, h, k)
plot_prediction_grid(xx,yy,prediction_grid, filename)

#uses built in knn function to classify the same iris
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(predictors, outcomes)
sk_predictions = knn.predict(predictors)

#makes observation with my knn classifier
my_predictions = np.array([knn_predict(p, predictors, outcomes, 5) for p in predictors])

#compare our knn classifier with the built in one
print("sk predictor and our homemade predictor agrees " + str(100 * np.mean(sk_predictions == my_predictions)) + "%")
print("sk predictor and actual outcome agrees " + str(100 * np.mean(sk_predictions == outcomes)) + "%")
print("our homemade predictor and actual outcome agrees " + str(100 * np.mean(my_predictions == outcomes)) + "%")
