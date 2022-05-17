import qrcode
a=input("whats ur name")
k=1
while k==1:
    try:
       b=int(input('age'))
       k=0
    except: k=1

q=[a,b]
qr=qrcode.make(q)
w=0
w=w+1
r='qr'+str(w)+'.png'
qr.save(r)