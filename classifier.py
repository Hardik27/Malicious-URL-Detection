import numpy as np
import pandas as pd
from scipy.stats import entropy
import re


data=pd.read_csv('data.csv')
data.head()

data['label'].value_counts()
data['label']=data['label'].replace({'bad':1,'good':0})
data['label']

data=data.dropna()
data['label']

data['length']=data['url'].str.len()

#Count the number of digits
def count_digits(string):
    return sum(item.isdigit() for item in string)

data['num_of_digits']=data['url'].apply(count_digits)

#first index of the digit in the url
def first_digit_index(s):
  m=re.search("\d",s)
  if m:
    return m.start()
  else:
    return -1

data['first_digit_index']=data['url'].apply(first_digit_index)


#vowel to consonant raito
def vowel_consonant_ratio(s):
  s=s.lower()
  vowel_pattern=re.compile('([aeiou])')
  consonants_pattern=re.compile('([b-df-hj-np-tv-z])')
  vowels=re.findall(vowel_pattern,s)
  consonants=re.findall(consonants_pattern,s)
  try:
    ratio=len(vowels)/len(consonants)
  except:
    ratio=0
  return ratio

data['ratio']=data['url'].apply(vowel_consonant_ratio)

#Number of semicolons, question marks, underscores, equals ampersands
def number_of_rare_characters(s):
  characters_pattern=re.compile('([;_?=&])')
  ch=re.findall(characters_pattern,s)
  return len(ch)

data['Number of Special Char']=data['url'].apply(number_of_rare_characters)

#digit to letter ratio
def digit_to_letter_ratio(s):
  alphabet=0
  digit=0
  for i in range(len(s)):
    if s[i].isalpha():
      alphabet=alphabet+1
    elif s[i].isdigit():
      digit=digit+1
  try:
    ratio=digit/alphabet
  except:
    ratio=0
  return ratio

data['Digit_To_Letter_Ratio']=data['url'].apply(digit_to_letter_ratio)

#Number of //
def number_of_double_slash(s):
  m=re.search("/",s)
  if m:
    start=m.start()
    path=s[start:]
    double_slashes=path.count("//")
    if double_slashes==None:
      double_slashes=0
    return double_slashes
  else:
    return 0

data['Number_of_Double_slashes']=data['url'].apply(number_of_double_slash)

#Presence of %20 in the path
def presence_of_20(s):
  m=re.search("/",s)
  if m:
    start=m.start()
    path=s[start:]
    if "%20" in path:
      return 1
    else:
      return 0

data['Presence_of_%20']=data['url'].apply(presence_of_20)
data['Presence_of_%20'].fillna(0,inplace=True)
#Number of 0 in the path
def number_of_zeroes(s):
  m=re.search("/",s)
  if m:
    start=m.start()
    path=s[start:]
    return path.count("0")
  else:
    return 0

data['Number_of_zeroes']=data['url'].apply(number_of_zeroes)

#Number of special character path 
def number_of_special_characters_in_path(s):
  m=re.search("/",s)
  special_char=0
  path=s
  if m:
    start=m.start()
    path=s[start:]
  for i in range(len(path)):
    if not path[i].isalpha() and not path[i].isdigit():
      special_char=special_char+1
  return special_char
  
data['Number_of_special_char_in_path']=data['url'].apply(number_of_special_characters_in_path)


#Primary domain operations
#Length of primary domain
def length_of_domain(s):
    m=re.search("/",s)
    domain=s
    if m:
        start=m.start()
        domain=s[:start]
    return len(domain)

data['Length_of_domain']=data['url'].apply(length_of_domain)

#Number of non-alphanumeric character in domain
def number_non_alphanumeric_in_domain(s):
    m=re.search("/",s)
    domain=s
    non_alphanumeric_char=0
    if m:
        start=m.start()
        domain=s[:start]
    for i in range(len(domain)):
        if not domain[i].isalpha() and not domain[i].isdigit():
            non_alphanumeric_char=non_alphanumeric_char+1
    return non_alphanumeric_char

data['Number_of_Non_alphanumeric_char_in_domain']=data['url'].apply(number_non_alphanumeric_in_domain)

#Number of hyphens in the domain
def number_of_hyphens_in_domain(s):
    m=re.search("/",s)
    domain=s
    hyphens=0
    if m:
        start=m.start()
        domain=s[:start]
    for i in range(len(domain)):
        if domain[i]=='-':
            hyphens=hyphens+1
    return hyphens

data['Number_of_hyphens_in_domain']=data['url'].apply(number_of_hyphens_in_domain)

#Number of @ in the domain
def number_of_atTheRate_in_domain(s):
    m=re.search("/",s)
    domain=s
    atTheRate=0
    if m:
        start=m.start()
        domain=s[:start]
    for i in range(len(domain)):
        if domain[i]=='@':
            atTheRate=atTheRate+1
    return atTheRate

data['Number_of_@_in_domain']=data['url'].apply(number_of_atTheRate_in_domain)

#If Domain contains ip
def contains_ip(s):
    m=re.search("/",s)
    domain=s
    if m:
        start=m.start()
        domain=s[:start]
    
    aa=re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",domain)
    if aa:
        return 1
    else:
        return 0
    
data['Domain_contains_ip']=data['url'].apply(contains_ip)

#f = open("alexa_top50.txt", "r")
#top_50=''
#top_50=f.read()
#print(f.read())

#domain present in top 50
def domain_present_in_top50_sites(s):
    m=re.search("/",s)
    domain=s
    if m:
        start=m.start()
        domain=s[:start]
    if domain in top_50:
        return 1
    else:
        return 0

#data['Domain_present_in_top50_sites']=data['url'].apply(domain_present_in_top50_sites)

from sklearn.model_selection import train_test_split
features=['length','num_of_digits','first_digit_index','ratio','Number of Special Char','Digit_To_Letter_Ratio','Number_of_Double_slashes','Presence_of_%20','Number_of_zeroes','Number_of_special_char_in_path','Length_of_domain','Number_of_Non_alphanumeric_char_in_domain','Number_of_hyphens_in_domain','Number_of_@_in_domain','Domain_contains_ip','Domain_present_in_top50_sites']
X=data[features]
y=data['label']

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.30, random_state=64)


#will take around 5 mins to train
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=250,random_state=62,max_features=16)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)

feature_imp = pd.Series(clf.feature_importances_).sort_values(ascending=False)
feature_imp

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
metrics.auc(fpr, tpr)

#svm Time pass not a good classifier
from sklearn.svm import SVC
svc=SVC(gamma='auto', max_iter=1000)
svc.fit(X_train,y_train)

y_svm_pred=svc.predict(X_test)

accuracy_score(y_test,y_svm_pred)
roc_auc_score(y_test, y_svm_pred)