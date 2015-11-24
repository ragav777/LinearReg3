__author__ = 'ragav777'
import numpy as np
import scipy.optimize as op
import csv
import math
import random


# CostFunction gets a X that is  m x (n+1) for theta0
# y is m x 1 theta is (n+1) x 1
def costfunction(theta, X, y, lda ):
    m,n = X.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    error = np.dot(X, theta) - y
    term1 = (1/(2*m)) * sum(np.square(error))
    temp2 = np.vstack((np.zeros((1,1), dtype = np.int), theta[1:n]))
    term2 = (lda/(2*m))*sum(np.square(temp2))
    J = term1 + term2
    return J

def gradient(theta, X, y, lda ):
    m,n = X.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    error = np.dot(X, theta) - y
    term1 = (1/m) * np.dot((X.T), error)
    temp2 = np.vstack((np.zeros((1,1), dtype = np.int), theta[1:n]))
    term2 = (lda/m)*temp2
    grad = term1 + term2
    return grad.flatten()

def trainlinearregression( X, y, lda, maxiter):
    m,n = X.shape
    initial_theta = np.zeros((n, 1))
    result = op.minimize(fun = costfunction, x0 = initial_theta, args = (X, y, lda), method = 'TNC',
             jac = gradient, options ={ 'disp': False, 'maxiter': maxiter }  )
    optimal_theta = result.x
    return optimal_theta

def cost_matrix(theta, X, y):
    m,n = X.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    error = np.dot(X, theta) - y
    rmsqe = math.sqrt(sum(np.square(error))/m)
    return error,rmsqe

def nfldataread(ppos) :
    with open('RB_5wkav.csv', 'r') as csvfile:
        nflreader = csv.reader(csvfile, delimiter=',')
        wfh = open ( (str(ppos)+ ".csv"), 'w', newline="")
        wfha = csv.writer(wfh)
        count = 0
        countr = 0
        playerindex= []

        ymap = { '2010':1/6, '2011':2/6, '2012':3/6, '2013':4/6, '2014':5/6, '2015':6/6 }
        tmap = { 'BAL':1/32, 'CIN':2/32, 'CLE':3/32, 'PIT':4/32, 'CHI':5/32, 'DET':6/32, 'GB':7/32, 'MIN':8/32, 'HOU':9/32, 'IND':10/32,
                 'JAC':11/32, 'TEN':12/32, 'ATL':13/32, 'CAR':14/32, 'NO':15/32, 'TB':16/32, 'BUF':17/32, 'MIA':18/32, 'NE':19/32,
                 'NYJ':20/32, 'DAL':21/32, 'NYG':22/32, 'PHI':23/32, 'WAS':24/32, 'DEN':25/32, 'KC':26/32, 'OAK':27/32, 'SD':28/32,
                 'ARI':29/32, 'SF':30/32, 'SEA':31/32, 'STL':32/32 }
        for row in nflreader:

            for i in [0,1,2,5,6,7,8,9,10,11,15,18,19,23,24,26,30,32,33,34,35,36,27,38] :
                if row[i] == '' :
                    row[i] = 0
                    #Zeroing out empty params

            if count != 0 :
                year = str(row[0]) #i
                game_eid = row[1] #i
                game_week = float(row[2])/17 #ni
                game_time = str(row[5]) #ni
                home_team = row[6] #i
                away_team = row[7] #i
                score_home = float(row[8])/50 #ni
                score_away= float(row[9])/50 #ni
                fumbles_tot = int(row[10])
                rushing_yards = float(row[11])/100 #o
                #receiving_lngtd = row[18]
                #rushing_twopta = row[25] #i
                rushing_tds = float(row[15])/3 #o
                #receiving_rec = row[34]
                #receiving_twopta = row[36]
                receiving_yds = float(row[18])/100
                rushing_att = float(row[19])/10 #i
                #reciving_twoptm = row[40] #o
                #rushing_lngtd = row[44]
                #receiving_lng = row[48]
                pos = row[23]
                receiving_tds = float(row[24])/3 #o
                name =row[26]
                #rushing_twoptm = row[61] #o
                #rushing_lng = row[64]
                team = row[30] #i
                points = float(row[32])
                n5wavyds = float(row[33])/100
                n5wavatt = float(row[34])/10
                n5wavrutd = float(row[35])/3
                n5wavnrec = float(row[36])/10
                n5wavrecyds = float(row[37])/100
                n5wavrectd = float(row[38])/3



                if ((pos == ppos)) :
                    if name not in playerindex :
                        playerindex.append(name)
                    player_number = float(playerindex.index(name)+1)/100

                    map_year = ymap[year] #m
                    if (team == home_team) :
                        playing_home = 1 #m
                    else :
                        playing_home = 0 #m

                    if (team == home_team) :
                        played_against = tmap[away_team] #m
                    else :
                        played_against = tmap[home_team] #m

                    #Added Game week #m
                    #Added Player team's score #m
                    if playing_home :
                        team_score = float(score_home) #nm
                    else :
                        team_score = float(score_away) #nm

                    #Added Opponent team's score #m
                    if playing_home :
                        opposition_score = float(score_away) #nm
                    else:
                        opposition_score = float(score_home) #nm

                    #(ghr,gmin) = game_time.split(":")
                    #time_played = int(ghr) + (int(gmin)/60) #nm

                    temp = str(game_eid)
                    month_played = float(int(temp[4:6])/12) #m

                    total_points = (((rushing_tds+ receiving_tds)*18) + (( rushing_yards +receiving_yds)*10) \
                                   -(fumbles_tot*2))/10

                    #temp = -4.3*(playerindex.index(name)+1) + (5.2*map_year) + (1.0033*playing_home)-(2.03*played_against) +(1*game_week)\
                            #+(2.5*game_week) + (1*time_played) + (2.6*team_score) + (3.2*opposition_score) + (1.6*month_played)

                    string = [ str(player_number), str(map_year), str(playing_home), str(played_against),
                               str(game_week), str(team_score), str(opposition_score),
                               str(month_played), str(tmap[team]),
                               str(n5wavyds), str(n5wavatt), str(n5wavrutd), str(n5wavnrec), str(n5wavrecyds),
                               str(n5wavrectd),
                               str(n5wavyds**2), str(n5wavatt**2), str(n5wavrutd**2), str(n5wavnrec**2), str(n5wavrecyds**2),
                               str(n5wavrectd**2),
                               str(total_points), str(name), str(points)  ]
                    wfha.writerow(string)
                    countr = countr +1
            count = count + 1
        wfh.close()
    return countr #Total records
    #print (countr) #Matched records

def createrandom(master,mtotal,mtrain,cv) :
    rndtemp = list(range(0,mtotal))
    random.shuffle(rndtemp)
    rndlinelist = sorted(rndtemp[1:mtrain+1])
    if cv:
        strng = "cv"
        wfh2 = open ( (master + "minus" + strng + ".csv"), 'w', newline="")
        wfha2 = csv.writer(wfh2)
    else:
        strng = "train"
    wfh = open ( (strng + str(mtrain) + ".csv"), 'w', newline="")
    wfha = csv.writer(wfh)
    count = 0
    countr = 0
    with open((str(master) + ".csv"), 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if count in rndlinelist:
                countr = countr +1 #matched
                wfha.writerow(row)
            elif ((count not in rndlinelist) and cv) :
                wfha2.writerow(row)

            count = count +1
    wfh.close()
    if cv:
        wfh2.close()
    return countr


def trainregression(mfile,lda,maxiter):
    Xtemp = np.loadtxt( (mfile +'.csv'), dtype = float, delimiter = ',', usecols = range(21) )
    mtr,ntr = np.shape(Xtemp)
    Xtrain = np.hstack ((np.ones ((mtr, 1)), Xtemp))
    Ytrain = np.loadtxt( (mfile + '.csv'), dtype = float, delimiter = ',', usecols = (21,) )
    theta = trainlinearregression( Xtrain, Ytrain, lda, maxiter)
    return theta


def cost_file(mfile, theta):
    Xtemp = np.loadtxt( (mfile +'.csv'), dtype = float, delimiter = ',', usecols = range(21) )
    m,n = np.shape(Xtemp)
    X = np.hstack ((np.ones ((m, 1)), Xtemp))
    y = np.loadtxt( (mfile + '.csv'), dtype = float, delimiter = ',', usecols = (21,) )
    theta = theta.reshape((n+1, 1))
    y = y.reshape((m, 1))
    error = np.dot(X, theta) - y
    # rmsqe = sum(abs(error))
    rmsqe = math.sqrt(sum(np.square(error))/m)
    return rmsqe

def main():

    lda = 0.001
    maxiter = 200
    master = 'RB'
    numcv = 1500
    iscv = 1
    istrain = 0

    mtotal = nfldataread(master) #returns num matched records
    print ("mtotal "+ str(mtotal))
    cvcount = createrandom(master,mtotal,numcv,iscv) #creates $master + "minuscv".csv and cv + $m.csv
    print ("cvcount "+ str(cvcount))
    mlist = [100,3467]
    for m in range(100,(mtotal-numcv),100):
    #for m in mlist :
        createrandom((master + "minus" + "cv"), (mtotal-numcv), m, istrain ) #creates $master + "minuscsv" + $m.csv
        theta = trainregression(("train" + str(m)),lda,maxiter)
        errortrain = cost_file(("train" + str(m)), theta)
        errorcv = cost_file(("cv"+str(cvcount)), theta)
        print ( "Trng m : " + str(m) + " Trng err : " + str(errortrain) + " cv error : " +  str(errorcv))
        print (theta)
if __name__ == '__main__' :
    main()
else :
    print("Didnt Work")
