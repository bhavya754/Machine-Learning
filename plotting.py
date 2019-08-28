import matplotlib.pyplot as plt
#Code to plot Demographic parity
n_groups=3
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.25
opacity = 0.8
vals_w=(dp_knn[0],dp_mle[0],dp_naive[0])
vals_b=(dp_knn[1],dp_mle[1],dp_naive[1])
rects1 = plt.bar(index, vals_w, bar_width,alpha=opacity, color='b', label='class 0')
rects2 = plt.bar(index + bar_width, vals_b, bar_width, alpha=opacity, color='g', label='class 1')
 
plt.xlabel('Classifier')
plt.ylabel('Values')
plt.title('DP')
plt.xticks(index + bar_width, ('knn', 'mle', 'naive'))
plt.legend()
plt.tight_layout()
plt.show()

#Code to plot Equalized odds
n_groups=3
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.25
opacity = 0.8
vals_w=(eo_knn[0]-eo_knn[1],eo_mle[0]-eo_mle[1],eo_naive[0]-eo_naive[1])
vals_b=(eo_knn[2]-eo_knn[3],eo_mle[2]-eo_mle[3],eo_naive[2]-eo_naive[3])
rects1 = plt.bar(index, vals_w, bar_width,alpha=opacity, color='b', label='True Positive Diff')
rects2 = plt.bar(index + bar_width, vals_b, bar_width, alpha=opacity, color='g', label='True Negative Diff')
  
plt.xlabel('Classifier')
plt.ylabel('Values')
plt.title('EO')
plt.xticks(index + bar_width, ('knn', 'mle', 'naive'))
plt.legend()
plt.tight_layout()
plt.show()

#Code to plot Predictive parity
n_groups=3
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.25
opacity = 0.8
vals_w=(pp_knn[0]-pp_knn[1],pp_mle[0]-pp_mle[1],pp_naive[0]-pp_naive[1])
vals_b=(pp_knn[2]-pp_knn[3],pp_mle[2]-pp_mle[3],pp_naive[2]-pp_naive[3])
rects1 = plt.bar(index, vals_w, bar_width,alpha=opacity, color='b', label='True Positive Diff')
rects2 = plt.bar(index + bar_width, vals_b, bar_width, alpha=opacity, color='g', label='True Negative Diff')
  
plt.xlabel('Classifier')
plt.ylabel('Values')
plt.title('PP')
plt.xticks(index + bar_width, ('knn', 'mle', 'naive'))
plt.legend()
plt.tight_layout()
plt.show()