#%%
####################################################################
#################################FUNCTIONS##########################
####################################################################
def ullhfit(x, fitfunc, limits=None, fixed=None, **kwargs):
    from iminuit import Minuit
    import numpy as np
    # Written with Philip Kofoed-Djursner
    def obt(*args):
        logf = np.zeros_like(x)
        
        # compute the function value
        f = fitfunc(x, *args)
    
        # find where the PDF is 0 or negative (unphysical)        
        mask_f_positive = f > 0

        # calculate the log of f everyhere where f is positive
        logf[mask_f_positive] = np.log(f[mask_f_positive])
        # set everywhere else to badvalue
        logf[~mask_f_positive] = -1000000
        
        # compute the sum of the log values: the LLH
        llh = -np.sum(logf)
        return llh

    ullh_Min = Minuit(obt, **kwargs, name = [*kwargs])
    ullh_Min.errordef = 0.5
    if limits:
        for i, limit in enumerate(limits):
            ullh_Min.limits[i] = limit
    if fixed:
        for i, fix in enumerate(fixed):
            ullh_Min.fixed[i] = fix
    ullh_Min.migrad()
    valuesfit = np.array(ullh_Min.values, dtype = np.float64)
    errorsfit = np.array(ullh_Min.errors, dtype = np.float64)
    if not ullh_Min.valid:
        print("!!! Fit did not converge !!!\n!!! Give better initial parameters !!!")
    # * Impliment p-value for ullh fit
    
    return valuesfit, errorsfit

def accept_reject(func, xmin, xmax, ymin, ymax, N_points, **kwargs):
    import numpy as np
    # Written with Philip Kofoed-Djursner
    # Recursive function to do accept/reject monte carlo simulation
    xran = np.random.uniform(xmin, xmax, N_points)
    yran = np.random.uniform(ymin, ymax, N_points)
    if isinstance(func, (tuple,list)):
        func_1 = func[0]
        func_2 = func[1]
        yfunc_1 = func_1(xran, **kwargs)
        yfunc_2 = func_2(xran, **kwargs)
        bool_mask = (yfunc_2 <= yran) & (yran <= yfunc_1)
    else:
        yfunc = func(xran, **kwargs)
        bool_mask = yran <= yfunc
    xkeep = xran[bool_mask]
    ykeep = yran[bool_mask] 
    missing = N_points - np.sum(bool_mask)
    if missing > 0:
        xrest, yrest, all_xrest, all_yrest, tries, bool_rest = accept_reject(func, xmin, xmax, ymin, ymax, missing, **kwargs)
    else:
        xrest = np.array([])
        yrest = np.array([])
        all_xrest = np.array([])
        all_yrest = np.array([])
        tries = 0
        bool_rest = np.array([], dtype=bool)
    finalx = np.append(xkeep, xrest)
    finaly = np.append(ykeep, yrest)
    allx = np.append(xran, all_xrest)
    ally = np.append(yran, all_yrest)
    final_bool = np.append(bool_mask, bool_rest)
    finaltries = N_points + tries
    return finalx, finaly, allx, ally, finaltries, final_bool

def accept_reject_df(func, xmin, xmax, ymin, ymax, N_points,x_name='x',y_name='y', **kwargs):
    import pandas as pd
    x, y, x_all, y_all, tries, bool_mask = accept_reject(func, xmin, xmax, ymin, ymax, N_points, **kwargs)
    df = pd.DataFrame(data={f'{x_name}':x_all,f'{y_name}':y_all,'accept':bool_mask})
    eff = N_points/tries
    area = (xmax - xmin) * (ymax - ymin) * eff
    return df, {'eff':eff,'area':area,'tries':tries}, x, y

def error_prop(vals, sigmas):
    from sympy import symbols, lambdify
    import numpy as np
    # Written by Ludvig Marcussen
    xSym, ySym = symbols("x y")
    symlist = [xSym, ySym]
    
    z = xSym/ySym
    
    def variances(func, symbols, values, sigmas):
        variance = np.zeros(len(symbols))
        for idx, (symbol, sigma) in enumerate(zip(symbols,sigmas)):
            f = lambdify(symbols, func.diff(symbol)**2 * sigma **2)
            variance[idx] = f(*values)
        return variance
    Vz = variances(z, symlist, vals, sigmas)
    sigmaz = np.sqrt(np.sum(Vz))
    zvalue = lambdify(symlist, z)(*vals)
    
    return zvalue, sigmaz

def run_all_classifiers(train_data, train_answer, test_data=None, test_answer=None, blind_data=None, verbose=False):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import Perceptron
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.ensemble import AdaBoostClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import GradientBoostingClassifier
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import pandas as pd
    classifier_names = [DecisionTreeClassifier,SVC,KNeighborsClassifier, GaussianNB, 
                   LinearDiscriminantAnalysis, LogisticRegression, Perceptron, 
                   RandomForestClassifier, QuadraticDiscriminantAnalysis, 
                   AdaBoostClassifier, XGBClassifier, GradientBoostingClassifier]
    classifier_names_str = ['DecisionTreeClassifier','SVC','KNeighborsClassifier', 'GaussianNB', 
                   'LinearDiscriminantAnalysis', 'LogisticRegression', 'Perceptron', 
                   'RandomForestClassifier', 'QuadraticDiscriminantAnalysis', 
                   'AdaBoostClassifier', 'XGBClassifier', 'GradientBoostingClassifier']
    classifier_init = [x() for x in classifier_names]
    models = [x.fit(train_data, train_answer) for x in classifier_init]

    acc_train = [x.score(train_data, train_answer) for x in models]
    [print(f'Train accuracy of {name} is {acc}') for name, acc in zip(classifier_names_str, acc_train)]

    fig, ax = plt.subplots()
    ax.bar(classifier_names_str, acc_train, align='center', alpha=0.5, color='#368015', edgecolor='black')
    ax.set_xticks(classifier_names_str, classifier_names_str,rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy of different classifiers on train data')
    ax.set_ylim([np.min(acc_train)-0.1, np.max(acc_train)+0.02])

    fig.show()
    # if verbose:
    #     twoclass_outputs = [DecisionBoundaryDisplay.from_estimator(x) for x in models]
    #     print(twoclass_outputs)
    #     fig, axes = plt.subplots((3,4))
    #     for i, ax in enumerate(axes):
    #         df_train_result = pd.DataFrame({'Class':twoclass_outputs[i],'Revenue':train_answer})
    #         sns.histplot(data=df_train_result, x='Class',hue='Revenue', palette=['r','k'],alpha=1,ax=ax,multiple='stack')
    #         ax.vlines(x=0, ymin=ax.get_ylim()[0], ymax=ax.get_ylim[1]+0.05*ax.get_ylim[1], linestyle='--', label='Separator', color='k')
    #         ax.set_xlabel('Classification Score')
    #         ax.set_title(f'Stacked classification histogram of revenue\nTrain set - Accuracy = {acc_train[i]:.2%}')

    if type(test_data)!=type(None) and type(test_answer)!=type(None):
        acc_test = [x.score(test_data, test_answer) for x in models]
        [print(f'Test accuracy of {name} is {acc}') for name, acc in zip(classifier_names_str, acc_test)]

        fig, ax = plt.subplots()
        ax.bar(classifier_names_str, acc_test, align='center', alpha=0.5, color='#368015', edgecolor='black')
        ax.set_xticks(classifier_names_str, classifier_names_str,rotation=45, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy of different classifiers on test data')
        ax.set_ylim([np.min(acc_test)-0.1, np.max(acc_test)+0.02])
        fig.show()
        # if verbose:
        #     twoclass_outputs = [x.decision_function(test_data) for x in models]
        #     fig, axes = plt.subplots((3,4))
        #     for i, ax in enumerate(axes):
        #         df_test_result = pd.DataFrame({'Class':twoclass_outputs[i],'Revenue':test_answer})
        #         sns.histplot(data=df_test_result, x='Class',hue='Revenue', palette=['r','k'],alpha=1,ax=ax,multiple='stack')
        #         ax.vlines(x=0, ymin=ax.get_ylim()[0], ymax=ax.get_ylim[1]+0.05*ax.get_ylim[1], linestyle='--', label='Separator', color='k')
        #         ax.set_xlabel('Classification Score')
        #         ax.set_title(f'Stacked classification histogram of revenue\nTest set - Accuracy = {acc_test[i]:.2%}')

    # if type(blind_data)!=type(None):
    #     twoclass_outputs = [x.decision_function(blind_data) for x in models]
    #     true_false = [x.predict(blind_data) for x in models]
    #     fig, axes = plt.subplots((3,4))
    #     for i, ax in enumerate(axes):
    #         df_test_result = pd.DataFrame({'Class':twoclass_outputs[i], 'Revenue':true_false})
    #         sns.histplot(data=df_test_result, x='Class',hue='Revenue', palette=['r','k'],alpha=1,ax=ax,multiple='stack')
    #         ax.vlines(x=0, ymin=ax.get_ylim()[0], ymax=ax.get_ylim[1]+0.05*ax.get_ylim[1], linestyle='--', label='Separator', color='k')
    #         ax.set_xlabel('Classification Score')
    #         ax.set_title(f'Stacked classification histogram of revenue\nBlind set - Accuracy = ???')

    return acc_train, acc_test, classifier_names_str

def run_ada_boost(train_data, train_answer, test_data=None, test_answer=None, blind_data=None, n_est=50, learn_rate=0.1, verbose=False,save=False):
    from sklearn.ensemble import AdaBoostClassifier
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    #Init
    ada_boost = AdaBoostClassifier(n_estimators=n_est,learning_rate=learn_rate)
    model = ada_boost.fit(train_data,train_answer)

    #Train
    acc_train = model.score(train_data, train_answer)
    twoclass_output_train = model.decision_function(train_data)
    
    if verbose:
        print(f'Train accuracy: {acc_train}')
        fig, ax = plt.subplots()
        df_train_result = pd.DataFrame({'Class':twoclass_output_train,'Revenue':train_answer})
        sns.histplot(data=df_train_result, x='Class',hue='Revenue',alpha=1,ax=ax, multiple='dodge')
        ax.vlines(x=0, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyle='--', label='Separator', color='k')
        ax.set_xlabel('Classification Score')
        ax.set_title(f'Stacked classification histogram of revenue\nTrain set - Accuracy = {acc_train:.2%}')

    #Test
    if type(test_data)!=type(None) and type(test_answer)!=type(None):
        acc_test = model.score(test_data, test_answer)
        twoclass_output_test = model.decision_function(test_data)

        if verbose:
            print(f'Test accuracy: {acc_test}')
            fig, ax = plt.subplots()
            df_test_result = pd.DataFrame({'Class':twoclass_output_test,'Revenue':test_answer})
            sns.histplot(data=df_test_result, x='Class',hue='Revenue',alpha=1,ax=ax,multiple='dodge')
            ax.vlines(x=0, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyle='--', label='Separator', color='k')
            ax.set_xlabel('Classification Score')
            ax.set_title(f'Stacked classification histogram of revenue\nTest set - Accuracy = {acc_test:.2%}')

    #Blind
    if type(blind_data)!=type(None):
        twoclass_output_blind = model.decision_function(blind_data.drop(['ID'], axis=1))
        true_false_blind = model.predict(blind_data.drop(['ID'], axis=1))
        df_blind_guess = pd.DataFrame({'Class':twoclass_output_blind, 'Revenue':true_false_blind})
        
        if verbose:
            print(f'Blind accuracy: ???')
            fig, ax = plt.subplots()
            sns.histplot(data=df_blind_guess, x='Class', hue='Revenue',alpha=1,ax=ax)
            ax.vlines(x=0, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyle='--', label='Separator', color='k')
            ax.set_xlabel('Classification Score')
            ax.set_title('Classification histogram of revenue\nBlind set - Accuracy = ???')

        if save:
            df_rev_true = blind_data['ID'][true_false_blind.astype(bool)]
            df_rev_false = blind_data['ID'][~true_false_blind.astype(bool)]
            df_rev_true.to_csv(r'C:\Users\nanna\OneDrive\Skrivebord\AMAS2023-main\AMAS_casper\Exam\Exam_2023\wied.Problem4.RevenueTrue.txt', index=False, header=False)
            df_rev_false.to_csv(r'C:\Users\nanna\OneDrive\Skrivebord\AMAS2023-main\AMAS_casper\Exam\Exam_2023\wied.Problem4.RevenueFalse.txt', index=False, header=False)
   
    return acc_train, acc_test

def raster_scan_2D(x, func, variables, ranges, points = [100, 100], vmax=None, **kwargs):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Written by Philip Kofoed-Djursner
    v1_range = np.linspace(*ranges[0], points[0])
    v2_range = np.linspace(*ranges[1], points[1])
    mesh = np.meshgrid(v1_range, v2_range)
    
    arguments = {**{key:value for key, value in zip(variables, mesh)}, **kwargs}
    
    llh = np.zeros((points[0], points[1]))
    for value in x:
        llh -= 2*np.log(func(value, **arguments))
    # llh = np.sum([np.log(func(value, **arguments)) for value in x], axis = 0)
    
    if vmax:
        ax = sns.heatmap(llh, cmap="Spectral", vmax=np.min(llh)+vmax)

    else:
        ax = sns.heatmap(llh, cmap="Spectral")
    ticks = 21
    ax.set_xticks(np.linspace(0, points[0], ticks))
    ax.set_xticklabels(np.around(np.linspace(*ranges[0], ticks),1))
    ax.set_yticks(np.linspace(0, points[1], ticks))
    ax.set_yticklabels(np.around(np.linspace(*ranges[1], ticks),1))
    
    ax.set_xlabel(variables[0])
    ax.set_ylabel(variables[1])
    ax.set_title('Raster Scan')
    plt.gca().invert_yaxis()

def raster_scan_3D(x, func, variables, ranges, points = [100, 100, 100], **kwargs):
    import numpy as np
    v1_range = np.linspace(*ranges[0], points[0])
    v2_range = np.linspace(*ranges[1], points[1])
    v3_range = np.linspace(*ranges[2], points[2])
    mesh = np.meshgrid(v1_range, v2_range, v3_range)
    
    arguments = {**{key:value for key, value in zip(variables, mesh)}, **kwargs}
    
    llh = np.zeros((points[0], points[1], points[2]))
    for value in x:
        llh -= 2*np.log(func(value, **arguments))
    max_llh = np.where(llh==np.min(llh))
    flat_arg_1 = v1_range[max_llh[0]]
    flat_arg_2 = v2_range[max_llh[1]]
    flat_arg_3 = v3_range[max_llh[2]]

    return (flat_arg_1, flat_arg_2, flat_arg_3)

def autocorrelate(azi, zen, splits):
    import numpy as np
    N_tot = len(azi)
    norm = 1/(N_tot*(N_tot-1)) 
    az, ze = np.meshgrid(azi, zen)
    X_i = np.sin(ze)*np.cos(az)
    Y_i = np.sin(ze)*np.sin(az)
    Z_i = np.cos(ze)
    X_j = np.sin(ze.T)*np.cos(az.T)
    Y_j = np.sin(ze.T)*np.sin(az.T)
    Z_j = np.cos(ze.T)
    
    cos_phi = X_i*X_j + Y_i*Y_j + Z_i*Z_j
    auto_points = np.empty(splits)
    for i,cos_phi_test in enumerate(np.linspace(-1, 1, splits)):
        C_matrix = np.heaviside(cos_phi - cos_phi_test, 1)
        auto_points[i] = ((np.sum(C_matrix) - np.trace(C_matrix))*norm)
    return auto_points

def bootstrap(data, func, xmin, xmax, ymin, ymax, N_points, N_pseudo_trials, **args_func):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    llh_data = np.sum(np.log(func(data, **args_func)))
    llh_values = []
    for _ in range(N_pseudo_trials):
        x, y, _, _, _, _ = accept_reject(func, xmin, xmax, ymin, ymax, N_points, **args_func)
        llh = np.sum(np.log(func(x, **args_func)))
        llh_values.append(llh)
    mean = np.mean(llh_values)
    kde = stats.gaussian_kde(llh_values)
    if llh_data > mean:
        left = kde.integrate_box_1d(-np.inf, mean-np.abs(mean-llh_data))
        right = kde.integrate_box_1d(llh_data, np.inf)
    if llh_data <= mean:
        left = kde.integrate_box_1d(-np.inf, llh_data)
        right = kde.integrate_box_1d(mean+np.abs(llh_data-mean), np.inf)

    print(f'P-value is {left+right}')
    fig, ax = plt.subplots()
    sns.histplot(x=llh_values, ax=ax, stat='density')
    ax.plot(np.linspace(np.min(llh_values), np.max(llh_values), 1000), kde(np.linspace(np.min(llh_values), np.max(llh_values), 1000)))
    ax.vlines(mean, ymin=0, ymax=ax.get_ylim()[1], label='MC mean llh', color='k')
    ax.vlines(llh_data, ymin=0, ymax=ax.get_ylim()[1], label='Data llh', color='#7eb0d5')
    ax.legend()

def likelihood(x, func, **args_func):
    import numpy as np
    logf = np.zeros_like(x)
    
    # compute the function value
    f = func(x, **args_func)

    # find where the PDF is 0 or negative (unphysical)        
    mask_f_positive = f > 0

    # calculate the log of f everyhere where f is positive
    logf[mask_f_positive] = np.log(f[mask_f_positive])
    # set everywhere else to badvalue
    logf[~mask_f_positive] = -1000000
    
    # compute the sum of the log values: the LLH
    llh = -np.sum(logf)
    return llh
####################################################################
###########################STANDARD PDF#############################
####################################################################
def gaussian(x, mu, sigma):
    import numpy as np
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*(((x-mu)**2)/(sigma**2)))
def poisson(x,mu):
    import numpy as np
    return ((mu**x) * np.exp(-mu))/(np.math.factorial(x))
def binomial(k,n,p):
    import numpy as np
    return (np.math.factorial(n)/(np.math.factorial(k)*np.math.factorial(n-k))) * (p**k) * ((1-p)**(n-k))
# %%
