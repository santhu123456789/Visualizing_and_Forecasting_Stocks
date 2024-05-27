import smtplib,ssl,random
EMAIL= "ENTER EMAIL"
PASSWD= "ENTER PASSWORD"
def sendotp(email):
    otp=random.randint(0000,9999)
    msg=f"""
    The once time password is {otp}
    """
    sl=ssl.create_default_context()
    smt=smtplib.SMTP("smtp.gmail.com",587)
    smt.starttls(context=sl)
    smt.login(user=EMAIL,password=PASSWD)
    smt.sendmail(EMAIL,email,msg)
    smt.close()
    return str(otp)