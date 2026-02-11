from utils.preprocess import clean_email


sample_mail="<html><body>Click <a href='http://phish.com'>here</a> to reset your password!</body></html>"

cleaned=clean_email(sample_mail)

print("Og: ", sample_mail)
print("Cleaned Mail :" , cleaned)