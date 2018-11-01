import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

data = pd.read_csv('train_x.csv',encoding='big5')
train_Y = pd.read_csv('train_y.csv',encoding='big5')
train_Y = train_Y.rename(columns={"Y":"default_payment"})
data = pd.concat([data, train_Y],axis=1)
print(data.columns)
#features = quant + qual_Enc + logged + [output]
features = data.columns
corr = data[features].corr()
plt.subplots(figsize=(30,10))
sns.heatmap( corr, square=True, annot=True, fmt=".1f" ) 
#output = 'default_payment'
#
## Let's do a little EDA
#cols = [ f for f in data.columns if data.dtypes[ f ] != "object"]
#cols.remove( output )
#
#f = pd.melt( data, id_vars=output, value_vars=cols)
#g = sns.FacetGrid( f, hue=output, col="variable", col_wrap=5, sharex=False, sharey=False )
#g = g.map( sns.distplot, "value", kde=True).add_legend()
#
#def ChiSquaredTestOfIndependence( df, inputVar, Outcome_Category ):
#    # Useful to have this wrapped in a function
#    # The ChiSquaredTest of Independence - 
#    # has a null hypothesis: the OutcomeCategory is independent of the inputVar
#    # So we create a test-statistic which is a measure of the difference between 
#    # "expected" i.e. what we WOULD observe if the OutcomeCategory WAS independent of the inputVar
#    # "observed" i.e. what the data actually shows
#    # the p-value returned is the probability of seeing this test-statistic if the null-hypothesis is true
#    Outcome_Category_Table = df.groupby( Outcome_Category )[ Outcome_Category ].count().values
#    Outcome_Category_Ratios = Outcome_Category_Table / sum( Outcome_Category_Table )
#    possibleVals = df[inputVar].unique()
#    observed = []
#    expected = []
#    for possible in possibleVals:
#        countsInCategories = df[ df[ inputVar ] == possible ].groupby( Outcome_Category )[Outcome_Category].count().values
#        if( len(countsInCategories) != len( Outcome_Category_Ratios ) ):
#            print("Error! The class " + str( possible) +" of \'" + inputVar + "\' does not contain all values of \'" + Outcome_Category + "\'" )
#            return
#        elif( min(countsInCategories) < 5 ):
#            print("Chi Squared Test needs at least 5 observations in each cell!")
#            print( inputVar + "=" + str(possible) + " has insufficient data")
#            print( countsInCategories )
#            return
#        else:
#            observed.append( countsInCategories )   
#            expected.append( Outcome_Category_Ratios * len( df[df[ inputVar ] == possible ]))
#    observed = np.array( observed )
#    expected = np.array( expected )
#    chi_squared_stat = ((observed - expected)**2 / expected).sum().sum()
#    degOfF = (observed.shape[0] - 1 ) *(observed.shape[1] - 1 ) 
#    #crit = stats.chi2.ppf(q = 0.95,df = degOfF) 
#    p_value = 1 - stats.chi2.cdf(x=chi_squared_stat, df=degOfF)
#    print("Calculated test-statistic is %.2f" % chi_squared_stat )
#    print("If " + Outcome_Category + " is indep of " + inputVar + ", this has prob %.2e of occurring" % p_value )
#    #t_stat, p_val, doF, expArray = stats.chi2_contingency(observed= observed, correction=False)
#    #print("Using built-in stats test: outputs")
#    #print("test-statistic=%.2f, p-value=%.2f, degsOfFreedom=%d" % ( t_stat, p_val, doF ) )
#    
##ChiSquaredTestOfIndependence( data, "SEX", output )
### Ok. So "default" is not independent of "SEX".
##ChiSquaredTestOfIndependence( data, "EDUCATION", output ) 
#print("We have %d with EDUCATION=0" % len(data.loc[ data["EDUCATION"]==0]))
#print("We have %d with EDUCATION=4" % len(data.loc[ data["EDUCATION"]==4]))
#print("We have %d with EDUCATION=5" % len(data.loc[ data["EDUCATION"]==5]))
#print("We have %d with EDUCATION=6" % len(data.loc[ data["EDUCATION"]==6]))
## Since we have 30k samples, let's just put these non-typical Education instances all into the EDUCATION=4 class and continue 
#data["EDUCATION_Corr"] = data["EDUCATION"].apply( lambda x: x if ((x>0) and (x<4)) else 4 )
#ChiSquaredTestOfIndependence( data, "EDUCATION_Corr", output ) 
#cols.remove("EDUCATION")
#cols.append("EDUCATION_Corr")
#
##ChiSquaredTestOfIndependence( data, "MARRIAGE", output ) 
## The quantitative vars:
#quant = ["LIMIT_BAL", "AGE"]
#
## The qualitative but "Encoded" variables (ie most of them)
#qual_Enc = cols
#qual_Enc.remove("LIMIT_BAL")
#qual_Enc.remove("AGE")
#logged = []
#for ii in range(1,7):
#    qual_Enc.remove("PAY_AMT" + str( ii ))
#    data[ "log_PAY_AMT" + str( ii )]  = data["PAY_AMT"  + str( ii )].apply( lambda x: np.log1p(x) if (x>0) else 0 )
#    logged.append("log_PAY_AMT" + str( ii ) )
#
#for ii in range(1,7):
#    qual_Enc.remove("BILL_AMT" + str( ii ))
#    data[ "log_BILL_AMT" + str( ii )] = data["BILL_AMT" + str( ii )].apply( lambda x: np.log1p(x) if (x>0) else 0 )
#    logged.append("log_BILL_AMT" + str( ii ) )
#
#f = pd.melt( data, id_vars=output, value_vars=logged)
#g = sns.FacetGrid( f, hue=output, col="variable", col_wrap=3, sharex=False, sharey=False )
#g = g.map( sns.distplot, "value", kde=True).add_legend()
 