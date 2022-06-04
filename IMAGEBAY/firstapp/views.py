from django.shortcuts import render, redirect
from firstapp import models
from firstapp.models import blogs
from django.conf import settings
from django.core.mail import send_mail
from firstapp.models import registeredUsers
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
# Create your views here.

def test(request):
	return render(request,'test.html')

def contact(request):
	if request.method=='POST':
		print('Form submitted')

		username=request.POST.get('user_name')
		email=request.POST.get('user_email')
		usermessage=request.POST.get('message')

		print(username,email,usermessage)

		c=models.enquiry() #create an object of the class

		c.username=username #obj.property=value
		c.emailid=email
		c.message=usermessage

		c.save()
		return render(request,'contactus.html',{'res':1})



	return render(request,'contactus.html')

def allblogs(request):
	blogs=models.blogs.objects.all()
	 #print(blogs)
	return render(request,'allblogs.html',{'data':blogs})

def forgotpassword(request):
	if(request.method=='POST'):
		em=request.POST.get('em')
		user=registeredUsers.objects.filter(emailid=em)

		if(len(user)>0):
			pw=user[0].password
			subject="Password"
			message="your Password is " + pw
			email_from=settings.EMAIL_HOST_USER
			recipient_list=[em,]
			send_mail(subject,message,email_from,recipient_list)
			rest="your password is sent to your respective email account. Please check"
			return render(request,'forgotpassword.html',{'rest':rest})
		else:
			res="This email id is not registered"
			return render(request,'forgotpassword.html',{'res':1})
	else:
		return render(request,'forgotpassword.html')



def userlogin(request):
    if request.method=='POST':
    	email=request.POST.get('user_mail')
    	password=request.POST.get('user_password')
    	print(email,password)

    	if models.registeredUsers.objects.filter(emailid=email,password=password).exists():
    		print('Logged in')
    		users=models.registeredUsers.objects.get(emailid=email)
    		request.session['userid']=users.id
    		return redirect('/dashboard')
    	else:
    		return render(request,'userlogin.html',{'res':1})
    return render(request,'userlogin.html')




def userregister(request):
	if request.method =='POST':
		print('Form submitted')

		username=request.POST.get('username')
		email=request.POST.get('email')
		password=request.POST.get('password')
		confirmpassword=request.POST.get('cpassword')

		print(username,email,password,confirmpassword)

		if password==confirmpassword:
			if models.registeredUsers.objects.filter(emailid=email).exists():
				return render(request,'userregister.html',{'res':2})
			else:
			   print('Registered')

			   user=models.registeredUsers()
			   user.username=username
			   user.emailid=email
			   user.password=password

			   user.save()
			   return render(request, 'userregister.html', {'res':3})




			   #return redirect('userlogin')
		else:
		    return render(request,'userregister.html',{'res':1})	



		
		


    
	return render(request,'userregister.html')

def index(request):
	return render(request,'index.html')

def home(request):
	return render(request,'home.html')

def readfullblog(request,id):
	blog=models.blogs.objects.get(id=id) #object
	print(blog)
	return render(request,'readfullblog.html' ,{'data':blog})

def userprofile(request):
	if not request.session.has_key('userid'):
		return redirect('userlogin')
	uid=request.session['userid']
	user=models.registeredUsers.objects.get(id=uid)
	return render(request,'userprofile.html', {'user':user})

def addreviews(request):
	if not request.session.has_key('userid'):
		return redirect('userlogin')
	uid=request.session['userid']
	user=models.registeredUsers.objects.get(id=uid)
	if request.method=='POST':
		r=models.reviews()
		r.subject=request.POST.get('subject')
		r.message=request.POST.get('review')
		r.user=user
		r.save()
		return render(request,'addreviews.html', {'res':1})	
	return render(request,'addreviews.html')	


def editprofile(request):
	if not request.session.has_key('userid'):
		return redirect('userlogin')
	uid=request.session['userid']
	user=models.registeredUsers.objects.get(id=uid)
	if request.method == 'POST':
		detail = models.registeredUsers.objects.get(id=uid)
		detail.username = request.POST.get('nm')
		detail.phoneNumber = request.POST.get('phn')
		detail.profilePicture = request.POST.get('pp')
		detail.dob = request.POST.get('db')
		detail.gender = request.POST.get('gn')
		detail.city = request.POST.get('ci')
		detail.state = request.POST.get('st')
		detail.save()
		data = models.registeredUsers.objects.get(id=uid)
		return render(request, 'userprofile.html' ,{'user':data})
	else:
		return render(request,'editprofile.html', {'user':user})	

def myprofile(request):
	return render(request,'myprofile.html')	

def changepassword(request):
	if request.method=='POST':
		reg=registeredUsers.objects.get(id=request.session['userid'])
		password=request.POST.get('opw')
		newpwd=request.POST.get('npw')
		confirmpwd=request.POST.get('cpw')
		if(newpwd==confirmpwd):
			p=reg.password
			if(password==p):
				reg.password=newpwd
				reg.confirmpassword=confirmpwd
				reg.save()
				rest="Password Changed !"
				return render(request,'changepassword.html',{'rest':rest})
			else:
				res="Invalid Current Password !"
				return render(request,'changepassword.html',{'res':res})
		else:
			res="Confirm Password and New Password Do Not Match"
			return render(request,'changePassword.html',{'res':res})
	else:
		return render(request,'changePassword.html')	
			       

		

def glossary(request):
	glossary=models.glossary.objects.all().order_by('word')
	return render(request,'glossary.html', {'glossary':glossary})	


def glossaryletter(request,id):
	glossary=models.glossary.objects.filter(word__istartswith=id)
	return render(request,'glossary.html', {'glossary':glossary})	

def logout(request):
    del request.session['userid']	
    return redirect('index')

def tutorial(request):
	tutorial=models.tutorials.objects.all
	return render(request,'tutorial.html',{'data':tutorial})
	
def handle_uploaded_file(f,name):
		destination = open(name, 'wb+')
		for chunk in f.chunks():
			destination.write(chunk)
			destination.close()

def Tools(request):
	if request.method=='POST':
		print("posted")
		from PIL import Image
		f = request.FILES['file1'] # here you get the files needed
		handle_uploaded_file(f,'STATIC/'+f.name)
		#'STATIC/temp1'+s[1]
		im = Image.open('STATIC/'+f.name)
		fm=im.format
		sz=im.size
		mode=im.mode
		nm=f.name
		print(im.format, im.size, im.mode)
		p=True
		f.name='STATIC/'+f.name
		q="filecontents"
		print("f.name",f.name)
		return render(request,'tools1.html',{'q':q,'nm':nm,'p':p,'fm':fm,'sz':sz,'mode':mode,'fname':f.name})
	else:
		p=False
		return render(request,'tools1.html',{'p':p})


		





def search_blog(request):
    x=True
    blog=request.POST.get("blog_name")	
    print(blog)

    b=blogs.objects.filter(title__contains=blog)
    print("b is", b)	
    return render(request,'allblogs.html',{'data':b})

def news(request):
	import datetime
	from datetime import date
	from newsapi.newsapi_client import NewsApiClient
	newsapi = NewsApiClient(api_key='cdb452a071a94eb6b8bd49d1c7bc87c4')
	json_data = newsapi.get_everything(q='image processing',
		                           language='en',
                                   from_param=str(date.today() - datetime.timedelta(days=29)),
                                   to=str(date.today()),
                                   
                                   page_size=18,
                                   page = 1,
                                   sort_by='relevancy')
	k=json_data['articles']
    
	return render(request,'news.html',{'k': k})	

def cropimage(request):
	if request.method=='POST':
		le=int(request.POST.get("le"))
		up=int(request.POST.get("up"))
		ri=int(request.POST.get("ri"))
		lo=int(request.POST.get("lo"))	
		print(le,up,ri,lo)
		if ri>le:			
		        
				import os
				
				
				f = request.FILES['file1'] # here you get the files needed
				import os
				s = os.path.splitext(f.name)
				print(s[1])
				handle_uploaded_file(f,'temp'+s[1])
				
				from PIL import Image
				im = Image.open('temp'+s[1]) 
				box = (le, up, ri, lo)
				region = im.crop(box)
				region.save('static/crop'+s[1]) 
				p="crop"+s[1]
				print("p",p)
				return render(request,'resultpage.html',{'p':p})
		else:
			msg=("please enter correct co-ordinates")
			return render(request,'cropimage.html',{'msg':msg})

	else:
		msg=""
		return render(request,'cropimage.html',{'msg':msg})

def resultpage(request):
	return render(request,'resultpage.html')

def transposeimage(request):
	if request.method=='POST':
		ttype=request.POST.get("ttype")
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		print("type",ttype)
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp'+s[1]) 

		if ttype=='FLIP_LEFT_RIGHT':
			print('lfr')
			out = im.transpose(Image.FLIP_LEFT_RIGHT)
			out.save('static/transpose'+s[1])
		if ttype=='FLIP_TOP_BOTTOM':
			out = im.transpose(Image.FLIP_TOP_BOTTOM)
			out.save('static/transpose'+s[1])
		if ttype=='ROTATE_90':
			out = im.transpose(Image.ROTATE_90)
			out.save('static/transpose'+s[1])
		if ttype=='ROTATE_180':
			out = im.transpose(Image.ROTATE_180)
			out.save('static/transpose'+s[1])
		if ttype=='ROTATE_270':
			out = im.transpose(Image.ROTATE_270)
			out.save('static/transpose'+s[1])
		
		p='transpose'+s[1]
		print("p")
		return render(request,'transposeresult.html',{'p':p})
		
	return render(request,'transposeimage.html')

def forgotpasswordemailsent(request):
	return render(request,'forgotpasswordemailsent.html')	

def blureffect(request):
	if request.method=='POST':
		ra=int(request.POST.get("ra"))
		print(ra)
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		print("radius",ra)
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp'+s[1])

		from PIL import ImageFilter
		OriImage = Image.open('temp'+s[1])
		OriImage.show()
		gaussImage = OriImage.filter(ImageFilter.GaussianBlur(ra))
		gaussImage.show()
		gaussImage.save('static/blur'+s[1])
		p='blur'+s[1]
		print("p")
		return render(request,'blurresult.html',{'p':p})
	else:
		return render(request,'blureffect.html')

	

def colorswap(request):
	return render(request,'colorswap.html')		

def resize(request):
	if request.method=='POST':
		wi=int(request.POST.get("wi"))
		he=int(request.POST.get("he"))
		print(wi,he)
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp'+s[1])
		sz=im.size
		nm=f.name
		print(im.size)

		import cv2
		img = cv2.imread('temp'+s[1], cv2.IMREAD_UNCHANGED)
		print('Original Dimensions : ',img.shape)
		width = wi
		height = he
		dim = (width, height)
		resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
		print('Resized Dimensions : ',resized.shape)
		cv2.imwrite('static/resize'+s[1],resized)
		p='resize'+s[1]
		print("p")
		return render(request,'resizeresult.html',{'sz':sz,'nm':nm,'p':p})
	else:
		return render(request,'resize.html')

def grayscale(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp'+s[1])

		
		img = Image.open('temp'+s[1])
		imgGray = img.convert('L')
		imgGray.save('static/grayscale'+s[1])
		p='grayscale'+s[1]
		print("p")
		return render(request,'grayscaleresult.html',{'p':p})
	else:
		return render(request,'grayscale.html')		

def facerecognition(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp'+s[1])

		import cv2
		import sys
		cascPath = "haarcascade_frontalface_default.xml"
		faceCascade = cv2.CascadeClassifier(cascPath)
		image = cv2.imread('temp'+s[1])
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(30, 30),
			flags = cv2.CASCADE_SCALE_IMAGE
			)
		print("Found {0} faces!".format(len(faces)))
		if len(faces)==0:
			msg="No face found! Please try another image"
			return render(request,'facerecognition.html',{'msg':msg})


		i=1
		for (x, y, w, h) in faces:
			cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
			cv2.imshow("Faces found", image)
		
		cv2.imwrite('static/face.jpg',image)
			
			
		print("p")
		return render(request,'facerecognitionresult.html',{'p':'static/face.jpg'})
	else:
		return render(request,'facerecognition.html')

def facerecognitionresult(request):
	return render(request,'facerecognitionresult.html')	

def addlogo(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])

		f2 = request.FILES['file2'] # here you get the files needed
		import os
		s2 = os.path.splitext(f2.name)
		print(s[1])
		handle_uploaded_file(f2,'temp2'+s[1])
		

		from PIL import Image, ImageDraw, ImageFilter
		im1 = Image.open('temp'+s[1])
		im2 = Image.open('temp2'+s[1])
		resized_im = im2.resize((round(im1.size[0]*0.25), round(im1.size[1]*0.25)))
		resized_im.show()
		im1.paste(resized_im)
		im1.save('static/addlogo'+s[1], quality=95)
		p='addlogo'+s[1]
		print("p")
		return render(request,'addlogoresult.html',{'p':p})
	else:
		return render(request,'addlogo.html')

def addlogoresult(request):
	return render(request,'addlogoresult.html')		

def splitmerge(request):
	return render(request,'splitmerge.html')


def imagevideo(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'STATIC/imagevideo/temp'+s[1])
		

		f = request.FILES['file2'] # here you get the files needed
		import os
		s1 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'STATIC/imagevideo/temp1'+s1[1])
		

		f = request.FILES['file3'] # here you get the files needed
		import os
		s2 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'STATIC/imagevideo/temp2'+s[1])
		

		f = request.FILES['file4'] # here you get the files needed
		import os
		s3 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'STATIC/imagevideo/temp3'+s[1])
		

		f = request.FILES['file5'] # here you get the files needed
		import os
		s4 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'STATIC/imagevideo/temp4'+s[1])
		

		import cv2
		import numpy as np
		import glob
		from PIL import Image
		frameSize = (500, 500)
		out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize)
		mean_height = 0
		mean_width = 0
		temp=Image.open('STATIC/imagevideo/temp'+s[1])
		width, height = temp.size
		mean_width += width
		mean_height += height
		temp1=Image.open('STATIC/imagevideo/temp1'+s1[1])
		width, height = temp1.size
		mean_width += width
		mean_height += height
		temp2=Image.open('STATIC/imagevideo/temp2'+s2[1])
		width, height = temp2.size
		mean_width += width
		mean_height += height
		temp3=Image.open('STATIC/imagevideo/temp3'+s3[1])
		width, height = temp3.size
		mean_width += width
		mean_height += height
		temp4=Image.open('STATIC/imagevideo/temp4'+s4[1])
		width, height = temp4.size
		mean_width += width
		mean_height += height
		mean_width=int(mean_width/5)
		mean_height=int(mean_height/5)
		im=Image.open('STATIC/imagevideo/temp'+s[1])
		imResize = im.resize((mean_width, mean_height), Image.ANTIALIAS)
		imResize.save('STATIC/imagevideo/retemp'+s[1], quality = 95) #
		im=Image.open('STATIC/imagevideo/temp1'+s[1])
		imResize = im.resize((mean_width, mean_height), Image.ANTIALIAS)
		imResize.save( 'STATIC/imagevideo/retemp1'+s1[1], quality = 95) #
		im=Image.open('STATIC/imagevideo/temp2'+s1[1])
		imResize = im.resize((mean_width, mean_height), Image.ANTIALIAS)
		imResize.save( 'STATIC/imagevideo/retemp2'+s2[1], quality = 95) #
		im=Image.open('STATIC/imagevideo/temp3'+s2[1])
		imResize = im.resize((mean_width, mean_height), Image.ANTIALIAS)
		imResize.save('STATIC/imagevideo/retemp3'+s3[1], quality = 95) #
		im=Image.open('STATIC/imagevideo/temp4'+s3[1])
		imResize = im.resize((mean_width, mean_height), Image.ANTIALIAS)
		imResize.save( 'STATIC/imagevideo/retemp4'+s4[1], quality = 95) #
		frame = cv2.imread('STATIC/imagevideo/retemp4'+s4[1])
		height, width, layers = frame.shape
		video = cv2.VideoWriter('STATIC/imagetovid.mp4', 0, 1, (width, height))
		video.write(cv2.imread('STATIC/imagevideo/retemp'+s[1]))
		video.write(cv2.imread('STATIC/imagevideo/retemp1'+s1[1]))
		video.write(cv2.imread('STATIC/imagevideo/retemp2'+s2[1]))
		video.write(cv2.imread('STATIC/imagevideo/retemp3'+s3[1]))
		video.write(cv2.imread('STATIC/imagevideo/retemp4'+s4[1]))
		video.release()  # releasing the video generated
		
		return render(request,'imagevideoresult.html')
	else:
		return render(request,'imagevideo.html')

def imagevideoresult(request):
	return render(request,'imagevideoresult.html')

def watermark(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		im1 = Image.open('temp'+s[1])
		width, height = im1.size

		f = request.FILES['file2'] # here you get the files needed
		import os
		s2 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp2'+s2[1])
		im2 = Image.open('temp2'+s2[1])
		im2 = im2.resize((round(im1.size[0]*0.25), round(im1.size[1]*0.25)))
		im2.save('watermark'+s[1])
		width2, height2 = im2.size
		if width<width2 or height<height2:
			msg="Logo is larger than the base image"
			return render(request,'watermark.html',{'msg':msg})	

		import cv2
		img = cv2.imread('temp'+s[1])
		watermark = cv2.imread('watermark'+s[1])
		percent_of_scaling = 20
		new_width = int(img.shape[1] * percent_of_scaling/100)
		new_height = int(img.shape[0] * percent_of_scaling/100)
		new_dim = (new_width, new_height)
		resized_img = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)
		wm_scale = 40
		wm_width = int(watermark.shape[1] * wm_scale/100)
		wm_height = int(watermark.shape[0] * wm_scale/100)
		wm_dim = (wm_width, wm_height)
		resized_wm = cv2.resize(watermark, wm_dim, interpolation=cv2.INTER_AREA)
		h_img, w_img, _ = resized_img.shape
		center_y = int(h_img/2)
		center_x = int(w_img/2)
		h_wm, w_wm, _ = resized_wm.shape
		top_y = center_y - int(h_wm/2)
		left_x = center_x - int(w_wm/2)
		bottom_y = top_y + h_wm
		right_x = left_x + w_wm
		roi = resized_img[top_y:bottom_y, left_x:right_x]
		result = cv2.addWeighted(roi, 1, resized_wm, 0.3, 0)
		resized_img[top_y:bottom_y, left_x:right_x] = result
		filename = 'Watermarked_Image.jpg'
		cv2.imwrite('static/watermark'+s[1],resized_img)
		p='watermark'+s[1]
		print("p")
		return render(request,'watermarkresult.html',{'p':p})
	else:
		return render(request,'watermark.html')	

def watermarkresult(request):
	return render(request,'watermarkresult.html')	

def cartoonimage(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp'+s[1])
		import cv2
		import numpy as np
		img = cv2.imread('temp'+s[1])
		def cartoonize(img, k):
			 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			 edges  = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8)
			 data = np.float32(img).reshape((-1, 3))
			 print("shape of input data: ", img.shape)
			 print('shape of resized data', data.shape)
			 criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
			 _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
			 center = np.uint8(center)
			 result = center[label.flatten()]
			 result = result.reshape(img.shape)
			 blurred = cv2.medianBlur(result, 3)
			 cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
			 return cartoon
		cartoonized = cartoonize(img, 8)
		cv2.imwrite('static/cartoon'+s[1],cartoonized)
		p='cartoon'+s[1]
		print("p")
		return render(request,'cartoonresult.html',{'p':p})
	else:
		return render(request,'cartoonimage.html')
		
		
			
def filters(request):
	return render(request,'filters.html')	

def blend(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp'+s[1])

		f2 = request.FILES['file2'] # here you get the files needed
		import os
		s2 = os.path.splitext(f2.name)
		print(s[1])
		handle_uploaded_file(f2,'temp2'+s[1])
		im2 = Image.open('temp2'+s[1])

		import cv2
		bg = cv2.imread('temp2'+s[1], cv2.IMREAD_COLOR)
		fg = cv2.imread('temp'+s[1], cv2.IMREAD_COLOR)
		dim = (1200, 800)
		resized_bg = cv2.resize(bg, dim, interpolation = cv2.INTER_AREA)
		resized_fg = cv2.resize(fg, dim, interpolation = cv2.INTER_AREA)
		blend = cv2.addWeighted(resized_bg, 0.5, resized_fg, 0.8, 0.0)
		cv2.imwrite('static/blend'+s[1], blend)
		p='blend'+s[1]
		print("p")
		return render(request,'blendresult.html',{'p':p})
	else:
		return render(request,'blend.html')	

def blendresult(request):
	return render(request,'blendresult.html')	

def analysis(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])

		from matplotlib import pyplot as plt
		from skimage import data
		from skimage.feature import blob_dog, blob_log, blob_doh
		from math import sqrt
		from skimage.color import rgb2gray
		import glob
		from skimage.io import imread
		from matplotlib import cm

		example_file = glob.glob(r'temp'+s[1])[0]
		im = imread(example_file, as_gray=True)
		blobs_log = blob_log(im, max_sigma=30, num_sigma=10, threshold=.1)
		blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
		numrows = len(blobs_log)
		print("Number of stars counted : " ,numrows)
		sa=numrows
		print(numrows)
		p=True
		return render(request,'analysis.html',{'sa':sa ,'p':p})
	else:
		p=False
		return render(request,'analysis.html', {'p':p})

def textimage(request):
	if request.method=="POST":
		def getSize(txt, font):
			testImg = Image.new('RGB', (1, 1))
			testDraw = ImageDraw.Draw(testImg)
			return testDraw.textsize(txt, font)
		fontname="Arial.ttf"
		fontsize = 11
		text = request.POST.get('t1')
		colorText = "black"
		colorOutline = "red"
		colorBackground = "white"
		font = ImageFont.truetype("arial.ttf", 15)
		width, height = getSize(text, font)
		img = Image.new('RGB', (width+20, height+20), colorBackground)
		d = ImageDraw.Draw(img)
		d.text((2, height/2), text, fill=colorText, font=font)
		img.save('static/textimage.png')
		p='textimage.png'
		print("p")
		return render(request,'textimageresult.html',{'p':p})
	else:
		return render(request,'textimage.html')	


def textimageresult(request):
	return render(request,'textimageresult.html')

def textextract(request):
	return render(request,'textextract.html')

def classification(request):
	return render(request,'classification.html')

def detection(request):
	return render(request,'detection.html')

def gif(request):
	import pointillism as pt
	if request.method=='POST':
		print("ok")
		f = request.FILES['file1']

		handle_uploaded_file(f,"g1.jpg")
		point = pt.image(location='g1.jpg', debug = True)
		point.crop([1000,500], False)
		point.display(original=True)
		point.make('balanced')
		point.display()
		point.save_out(location='static', suffix='basic test')
		point.colormap('cyanotype')
		point._newImage(border=100)
		point.make('fine')
		point.display()
		point.colormap('sepia')
		point._newImage(border=100)
		point.make('fine')
		point.display()
		point = pt.pipeline(location='g1.jpg', debug = True, border = 0)
		point.make_gif(kind='assembly', location='static', name='g1.gif', crop=True)
		p='g1.gif'
		return render(request,'gifresult.html',{'p':p})
	else:
		return render(request,'gif.html')

def gifresult(request):
	return render(request,'gifresult.html')

def transposeresult(request):
	return render(request,'transposeresult.html')

def blurresult(request):
	return render(request,'blurresult.html')

def cartoonresult(request):
	return render(request,'cartoonresult.html')

def resizeresult(request):
	return render(request,'resizeresult.html')

def grayscaleresult(request):
	return render(request,'grayscaleresult.html')

def contour(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp'+s[1])

		from PIL import ImageFilter
		from PIL.ImageFilter import (
			BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
			EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
			)
		img = Image.open('temp'+s[1])
		img1 = img.filter(CONTOUR)
		img1.save('static/contour'+s[1])
		p='contour'+s[1]
		print("p")
		return render(request,'contourresult.html',{'p':p})
	else:
		return render(request,'contour.html')

def contourresult(request):
	return render(request,'contourresult.html')

def edgeenhance(request):
	if request.method=='POST':
		etype=request.POST.get("etype")
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp'+s[1]) 

		from PIL.ImageFilter import (
			BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
			EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
			)
		img1 = im.filter(EDGE_ENHANCE_MORE)
		img1.save('static/edgeenhance'+s[1])
		p='edgeenhance'+s[1]
		print("p")
		return render(request,'edgeenhanceresult.html',{'p':p})
	else:
		return render(request,'edgeenhance.html')

def edgeenhanceresult(request):
	return render(request,'edgeenhanceresult.html')

def emboss(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp'+s[1])

		from PIL import ImageFilter
		from PIL.ImageFilter import (
			BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
			EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
			)
		img = Image.open('temp'+s[1])
		img1 = img.filter(EMBOSS)
		img1.save('static/emboss'+s[1])
		p='emboss'+s[1]
		print("p")
		return render(request,'embosseffect.html',{'p':p})
	else:
	    return render(request,'emboss.html')

def embosseffect(request):
	return render(request,'embosseffect.html')

def findedge(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp'+s[1])

		from PIL import ImageFilter
		from PIL.ImageFilter import (
			BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
			EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
			)
		img = Image.open('temp'+s[1])
		img1 = img.filter(FIND_EDGES)
		img1.save('static/findedges'+s[1])
		p='findedges'+s[1]
		print("p")
		return render(request,'findedgeresult.html',{'p':p})
	else:
		return render(request,'findedge.html')

def findedgeresult(request):
	return render(request,'findedgeresult.html')

def smooth(request):
	if request.method=='POST':
		stype=request.POST.get("stype")
		print(stype)
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp'+s[1]) 

		from PIL.ImageFilter import (
			BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
			EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
			)
		
		img1 = im.filter(SMOOTH_MORE)
		img1.save('static/smooth'+s[1])
		p='smooth'+s[1]
		print("p")
		return render(request,'smoothresult.html',{'p':p})
	else:
		return render(request,'smooth.html')

def smoothresult(request):
	return render(request,'smoothresult.html')

def sharpen(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp'+s[1])

		from PIL import ImageFilter
		from PIL.ImageFilter import (
			BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
			EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
			)
		img = Image.open('temp'+s[1])
		img1 = img.filter(SHARPEN)
		img1.save('static/sharpen'+s[1])
		p='sharpen'+s[1]
		print("p")
		return render(request,'sharpenresult.html',{'p':p})
	else:
	    return render(request,'sharpen.html')

def sharpenresult(request):
	return render(request,'sharpenrresult.html')

def sepia(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp'+s[1])

		import cv2
		import numpy as np
		import scipy
		image = cv2.imread('temp'+s[1])
		def sepia(img):
			img_sepia = np.array(img, dtype=np.float64)
			img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
				[0.349, 0.686, 0.168],
				[0.393, 0.769, 0.189]]))
			img_sepia[np.where(img_sepia > 255)] = 255 
			img_sepia = np.array(img_sepia, dtype=np.uint8)
			return img_sepia
		a5 = sepia(image)
		filename = 'static/sepiafilter'+s[1]
		cv2.imwrite(filename, a5)
		p='sepiafilter'+s[1]
		print("p")
		return render(request,'sepiaresult.html',{'p':p})
	else:
		return render(request,'sepia.html')

def sepiaresult(request):
	return render(request,'sepiaresult.html')

def pencil(request):
	if request.method=='POST':
		ptype=request.POST.get("ptype")
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp'+s[1]) 

		import cv2
		import numpy as np
		import scipy
		image = cv2.imread('temp'+s[1])
		def pencil_sketch_col(img):
			sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
			return  sk_color
		a6 = pencil_sketch_col(image)
		filename = 'static/pencil'+s[1]
		cv2.imwrite(filename, a6)
		p='pencil'+s[1]
		print("p")
		return render(request,'pencilresult.html',{'p':p})
	else:
		return render(request,'pencil.html')

def pencilresult(request):
	return render(request,'pencilresult.html')

def HDR(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp'+s[1])

		import cv2
		import numpy as np
		import scipy
		image = cv2.imread('temp'+s[1])
		def HDR(img):
			hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
			return  hdr
		a8 = HDR(image)
		filename = 'static/HDR'+s[1]
		cv2.imwrite(filename,a8)
		p='HDR'+s[1]
		print("p")
		return render(request,'HDRresult.html',{'p':p})
	else:
		return render(request,'HDR.html')

def HDRresult(request):
	return render(request,'HDRresult.html')

def invert(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp'+s[1])

		import cv2
		import numpy as np
		import scipy
		image = cv2.imread('temp'+s[1])
		def invert(img):
			inv = cv2.bitwise_not(img)
			return inv
		a9 = invert(image)
		filename = 'static/invert'+s[1]
		cv2.imwrite(filename, a9)
		p='invert'+s[1]
		print("p")
		return render(request,'invertresult.html',{'p':p})
	else:
		return render(request,'invert.html')

def invertresult(request):
	return render(request,'invertresult.html')

def summer(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp'+s[1])
		import cv2
		import numpy as np
		import scipy
		from scipy.interpolate import UnivariateSpline
		image = cv2.imread('temp'+s[1])
		def LookupTable(x, y):
			spline = UnivariateSpline(x, y)
			return spline(range(256))
		def Summer(img):
				increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
				decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
				blue_channel, green_channel,red_channel  = cv2.split(img)
				red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
				blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
				sum= cv2.merge((blue_channel, green_channel, red_channel ))
				return sum
		a11 = Summer(image)
		filename = 'static/sunny'+s[1]
		cv2.imwrite(filename, a11)
		p='sunny'+s[1]
		print("p")
		return render(request,'summerresult.html',{'p':p})
	return render(request,'summer.html')

def summerresult(request):
    return render(request,'summerresult.html')

def winter(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp'+s[1])

		import cv2
		import numpy as np
		import scipy
		from scipy.interpolate import UnivariateSpline
		image = cv2.imread('temp'+s[1])
		def LookupTable(x, y):
			spline = UnivariateSpline(x, y)
			return spline(range(256))
		def Winter(img):
			increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
			decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
			blue_channel, green_channel,red_channel = cv2.split(img)
			red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
			blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
			win= cv2.merge((blue_channel, green_channel, red_channel))
			return win
		a10 = Winter(image)
		filename = 'static/winter'+s[1]
		cv2.imwrite(filename, a10)
		p='winter'+s[1]
		print("p")
		return render(request,'winterresult.html',{'p':p})
	else:
		return render(request,'winter.html')

def winterresult(request):
	return render(request,'winterresult.html')

def pngjpg(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp.png')

		print(im.mode)
		im.convert('RGB').save('static/pngtojpg.jpg')
		p='pngtojpg.jpg'
		print("p")
		return render(request,'pngjpgresult.html',{'p':p})
	return render(request,'pngjpg.html')

def pngjpgresult(request):
	return render(request,'pngjpgresult.html')

def imagetext(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])

		import pytesseract
		from PIL import Image
		import pyttsx3
		from googletrans import Translator
		img = Image.open('temp'+s[1])
		print(img)
		pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe' 
		result = pytesseract.image_to_string(img)
		with open('abc.txt',mode ='w') as file:
			file.write(result)
			print(result)
			p='temp'+s[1]
			print("p")
		return render(request,'imagetextresult.html',{'result':result})
	return render(request,'imagetext.html')

def imagetextresult(request):
	return render(request,'imagetextresult.html')

def movingtext(request):
    if request.method=='POST':
        t=request.POST.get("t1")
        f = request.FILES['file1']
        handle_uploaded_file(f,'movingtext2.jpg')        
        def create_image_with_text(size, text):
            global c, inc
            c = 0
            inc = 10
            img = Image.open("movingtext2.jpg")
            w,h = img.size
            draw = ImageDraw.Draw(img)
            draw.text((w // 5, h // 2), text, font = fnt, fill=(c,c,c))
            c += inc
            return img
 

        frames = []
 
        def roll(text):
             global c
             for i in range(len(text)+1):
                new_frame = create_image_with_text((0,0), text[:i])
                frames.append(new_frame)
             c = 0
 
        fnt = ImageFont.truetype("arial", 72)
        all_text = t.splitlines()
        for text in all_text:
            roll(text)
 
 

        frames[0].save('static/movingtext2.gif', format='GIF',
               append_images=frames[1:], save_all=True, duration=80, loop=0)
        print("Done")

        return  render (request,'movingtextresult.html')
    else :
        return  render (request,'movingtext.html')

def movingtextresult(request):
	return render(request,'movingtextresult.html')

def drawellipse(request):
    if request.method=='POST':
        t=request.POST.get("t1")    
        f = request.FILES['file1']
        handle_uploaded_file(f,'drawellipse.jpg')
        def create_image_with_text(wh, text):
            width, height = wh
            img = Image.open('drawellipse.jpg')
            draw = ImageDraw.Draw(img)
            fnt = ImageFont.truetype("arial", 72)        
            draw.text((width, height), text, font = fnt, fill="black")
            return img
        frames = []
        x, y = 0, 0
        for i in range(100):
            new_frame = create_image_with_text((x-100,y), t)
            frames.append(new_frame)
            x += 4
            y += 1
        frames[0].save('static/drawellipse.gif', format='GIF',append_images=frames[1:], save_all=True, duration=30, loop=10)

        return  render (request,'drawellipseresult.html')
    else :
        return  render (request,'drawellipse.html')




def drawellipseresult(request):
	return render(request,'drawellipseresult.html')

def brightness(request):
	if request.method=='POST':
		ti=int(request.POST.get("ti"))

		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])

		import PIL
		from PIL import ImageEnhance
		img = PIL.Image.open('temp'+s[1])
		converter = ImageEnhance.Brightness(img)
		img3 = converter.enhance(ti)
		img3.save('static/brightness'+s[1])
		p='brightness'+s[1]
		print("p")
		return render(request,'brightnessresult.html',{'p':p})
	else:
		return render(request,'brightness.html')

def brightnessresult(request):
	return render(request,'brightnessresult.html')

def contrast(request):
	if request.method=='POST':
		ti=int(request.POST.get("cc"))

		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])

		import PIL
		from PIL import ImageEnhance
		img = PIL.Image.open('temp'+s[1])
		converter = ImageEnhance.Contrast(img)
		img3 = converter.enhance(5)
		img3.save('static/contrast'+s[1])
		p='contrast'+s[1]
		print("p")
		return render(request,'contrastresult.html',{'p':p})
	else:
		return render(request,'contrast.html')

def contrastresult(request):
	return render(request,'contrastresult.html')

def saturation(request):
	if request.method=='POST':
		cs=int(request.POST.get('cs'))

		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])

		import PIL
		from PIL import ImageEnhance
		img = PIL.Image.open('temp'+s[1])
		converter = ImageEnhance.Color(img)
		img3 = converter.enhance(cs)
		img3.save('static/saturation'+s[1])
		p='saturation'+s[1]
		print("p")
		return render(request,'saturationresult.html',{'p':p})
	else:
		return render(request,'saturation.html')

def saturationresult(request):
	return render(request,'saturationresult.html')

def vignette(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp'+s[1])

		import numpy as np
		import cv2
		input_image = cv2.imread('temp'+s[1])
		input_image = cv2.resize(input_image, (480, 480))
		rows, cols = input_image.shape[:2]
		X_resultant_kernel = cv2.getGaussianKernel(cols,200)
		Y_resultant_kernel = cv2.getGaussianKernel(rows,200)
		resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T
		mask = 255 * resultant_kernel / np.linalg.norm(resultant_kernel)
		output = np.copy(input_image)
		for i in range(3):
			output[:,:,i] = output[:,:,i] * mask
			cv2.imshow('Original', input_image)
			cv2.imwrite('static/vignette'+s[1],output)
			p='vignette'+s[1]
			print("p")
			return render(request,'vignetteresult.html',{'p':p})
	else:
		return render(request,'vignette.html')

def vignetteresult(request):
	return render(request,'vignetteresult.html')

def adjust(request):
	if request.method=='POST':
		sa=int(request.POST.get("sa"))
		co=int(request.POST.get("co"))
		br=int(request.POST.get("br"))	
		print(sa,co,br)

		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])

		from fimage import FImage
		from fimage.filters import Contrast, Brightness, Saturation
		
		image = FImage('temp'+s[1])
		image.apply(
				Saturation(sa),
				Contrast(co),
				Brightness(br)
				)
		image.save('static/adjust'+s[1])
			
		p="adjust"+s[1]
		print("p",p)
		return render(request,'adjustresult.html',{'p':p})
	else:
		return render(request,'adjust.html')

def adjustresult(request):
	return render(request,'adjustresult.html')

def sincity(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])

		from fimage import FImage
		from fimage.presets import SinCity
		
		image = FImage('temp'+s[1])
		image.apply(SinCity())
		image.save('static/scfilter'+s[1])
		p='scfilter'+s[1]
		print("p")
		
		
		return render(request,'sincityresult.html',{'p':p})
	else:
		return render(request,'sincity.html')

def sincityresult(request):
	return render(request,'sincityresult.html')

def vintage(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp'+s[1])

		import cv2
		import numpy as np
		import matplotlib
		from matplotlib import pyplot as plt
		im = cv2.imread('temp'+s[1])
		rows, cols = im.shape[:2]
		kernel_x = cv2.getGaussianKernel(cols,200)
		kernel_y = cv2.getGaussianKernel(rows,200)
		kernel = kernel_y * kernel_x.T
		filter = 255 * kernel / np.linalg.norm(kernel)
		vintage_im = np.copy(im)
		for i in range(3):
			vintage_im[:,:,i] = vintage_im[:,:,i] * filter
		plt.imshow(vintage_im)
		plt.imsave('static/vintage'+s[1],vintage_im)
		p='vintage'+s[1]
		print("p")
		return render(request,'vintageresult.html',{'p':p})
	else:
		return render(request,'vintage.html')

def vintageresult(request):
	return render(request,'vintageresult.html')

def collage(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		i1='STATIC/temp'+s[1]
		handle_uploaded_file(f,'STATIC/temp'+s[1])
		

		f = request.FILES['file2'] # here you get the files needed
		import os
		s1 = os.path.splitext(f.name)
		print(s[1])
		i2='STATIC/temp1'+s[1]
		handle_uploaded_file(f,'STATIC/temp1'+s[1])
		

		f = request.FILES['file3'] # here you get the files needed
		import os
		s2 = os.path.splitext(f.name)
		print(s[1])
		i3='STATIC/temp2'+s[1]
		handle_uploaded_file(f,'STATIC/temp2'+s[1])
		

		f = request.FILES['file4'] # here you get the files needed
		import os
		s3 = os.path.splitext(f.name)
		print(s[1])
		i4='STATIC/temp3'+s[1]
		handle_uploaded_file(f,'STATIC/temp3'+s[1])

		import cv2
		import numpy as np
		image1=cv2.imread(i1)
		image2=cv2.imread(i2)
		image3=cv2.imread(i3)
		image4=cv2.imread(i4)
		image1=cv2.resize(image1,(200,200))
		image2=cv2.resize(image2,(200,200))
		image3=cv2.resize(image3,(200,200))
		image4=cv2.resize(image4,(200,200))
		Horizontal1=np.hstack([image1,image2])
		Horizontal2=np.hstack([image3,image4])
		Vertical_attachment=np.vstack([Horizontal1,Horizontal2])
		cv2.imwrite('static/finalcollage'+s[1],Vertical_attachment)
		p='finalcollage'+s[1]
		print("p")
		return render(request,'collageresult.html',{'p':p})
	else:
		return render(request,'collage.html')
	
def collageresult(request):
	return render(request,'collageresult.html')

def collage2(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp'+s[1])

		f2 = request.FILES['file2'] # here you get the files needed
		import os
		s2 = os.path.splitext(f2.name)
		print(s[1])
		handle_uploaded_file(f2,'temp2'+s[1])
		im2 = Image.open('temp2'+s[1])

		import cv2
		import numpy as np
		image1=cv2.imread('temp'+s[1])
		image2=cv2.imread('temp2'+s[1])
		image1=cv2.resize(image1,(200,200))
		image2=cv2.resize(image2,(200,200))
		Horizontal1=np.hstack([image1,image2])
		Vertical_attachment=np.vstack([Horizontal1])
		cv2.imwrite('STATIC/FinalCollage2'+s[1],Vertical_attachment)
		p='FinalCollage2'+s[1]
		print("p")
		return render(request,'collage2result.html',{'p':p})
	else:
		return render(request,'collage2.html')

def collage2result(request):
	return render(request,'collage2result.html')

def collage6(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp'+s[1])

		f = request.FILES['file2'] # here you get the files needed
		import os
		s2 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp2'+s2[1])
		im = Image.open('temp2'+s2[1])

		f = request.FILES['file3'] # here you get the files needed
		import os
		s3 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp3'+s3[1])
		im = Image.open('temp3'+s3[1])

		f = request.FILES['file4'] # here you get the files needed
		import os
		s4 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp4'+s4[1])
		im = Image.open('temp4'+s4[1])

		f = request.FILES['file5'] # here you get the files needed
		import os
		s5 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp5'+s5[1])
		im = Image.open('temp5'+s5[1])

		f = request.FILES['file6'] # here you get the files needed
		import os
		s6 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp6'+s6[1])
		im2 = Image.open('temp6'+s6[1])

		import cv2
		import numpy as np
		image1=cv2.imread('temp'+s[1])
		image2=cv2.imread('temp2'+s2[1])
		image3=cv2.imread('temp3'+s3[1])
		image4=cv2.imread('temp4'+s4[1])
		image5=cv2.imread('temp5'+s5[1])
		image6=cv2.imread('temp6'+s6[1])
		image1=cv2.resize(image1,(200,200))
		image2=cv2.resize(image2,(200,200))
		image3=cv2.resize(image3,(200,200))
		image4=cv2.resize(image4,(200,200))
		image5=cv2.resize(image5,(200,200))
		image6=cv2.resize(image6,(200,200))
		Horizontal1=np.hstack([image1,image2])
		Horizontal2=np.hstack([image3,image4])
		Horizontal3=np.hstack([image5,image6])
		Vertical_attachment=np.vstack([Horizontal1,Horizontal2,Horizontal3])
		cv2.imwrite('STATIC/FinalCollage6'+s[1],Vertical_attachment)
		p='FinalCollage6'+s[1]
		print("p")
		return render(request,'collageresult6.html',{'p':p})
	else:
		return render(request,'collage6.html')

def collageresult6(request):
	return render(request,'collageresult6.html')

def collage8(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		#im = Image.open('temp'+s[1])

		f = request.FILES['file2'] # here you get the files needed
		import os
		s2 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp2'+s2[1])
		#im2 = Image.open('temp2'+s[1])

		f = request.FILES['file3'] # here you get the files needed
		import os
		s3 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp3'+s3[1])
		#im3 = Image.open('temp3'+s[1])

		f = request.FILES['file4'] # here you get the files needed
		import os
		s4 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp4'+s4[1])
		#im4 = Image.open('temp4'+s[1])

		f = request.FILES['file5'] # here you get the files needed
		import os
		s5 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp5'+s5[1])
		#im5 = Image.open('temp5'+s5[1])

		f = request.FILES['file6'] # here you get the files needed
		import os
		s6 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp6'+s6[1])
		#im6 = Image.open('temp6'+s[1])


		f = request.FILES['file7'] # here you get the files needed
		import os
		s7 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp7'+s7[1])
		#im7 = Image.open('temp7'+s[1])

		f = request.FILES['file8'] # here you get the files needed
		import os
		s8 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp8'+s8[1])
		#im8 = Image.open('temp8'+s[1])

		import cv2
		import numpy as np
		image1=cv2.imread('temp'+s[1])
		image2=cv2.imread('temp2'+s2[1])
		image3=cv2.imread('temp3'+s3[1])
		image4=cv2.imread('temp4'+s4[1])
		image5=cv2.imread('temp5'+s5[1])
		image6=cv2.imread('temp6'+s6[1])
		image7=cv2.imread('temp7'+s7[1])
		image8=cv2.imread('temp8'+s8[1])
		image1=cv2.resize(image1,(200,200))
		image2=cv2.resize(image2,(200,200))
		image3=cv2.resize(image3,(200,200))
		image4=cv2.resize(image4,(200,200))
		image5=cv2.resize(image5,(200,200))
		image6=cv2.resize(image6,(200,200))
		image7=cv2.resize(image7,(200,200))
		image8=cv2.resize(image8,(200,200))
		Horizontal1=np.hstack([image1,image2])
		Horizontal2=np.hstack([image3,image4])
		Horizontal3=np.hstack([image5,image6])
		Horizontal4=np.hstack([image7,image8])
		Vertical_attachment=np.vstack([Horizontal1,Horizontal2,Horizontal3,Horizontal4])
		cv2.imwrite('STATIC/FinalCollage8'+s[1],Vertical_attachment)
		p='FinalCollage8'+s[1]
		print("p")
		return render(request,'collageresult8.html',{'p':p})
	else:
		return render(request,'collage8.html')

def collageresult8(request):
	return render(request,'collageresult8.html')

def collage10(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp1'+s[1])
		im = Image.open('temp1'+s[1])

		f = request.FILES['file2'] # here you get the files needed
		import os
		s2 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp2'+s2[1])
		im = Image.open('temp2'+s2[1])

		f = request.FILES['file3'] # here you get the files needed
		import os
		s3 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp3'+s3[1])
		im = Image.open('temp3'+s3[1])

		f = request.FILES['file4'] # here you get the files needed
		import os
		s4 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp4'+s4[1])
		im = Image.open('temp4'+s4[1])

		f = request.FILES['file5'] # here you get the files needed
		import os
		s5 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp5'+s5[1])
		im = Image.open('temp5'+s5[1])

		f= request.FILES['file6'] # here you get the files needed
		import os
		s6 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp6'+s6[1])
		im = Image.open('temp6'+s6[1])

		f = request.FILES['file7'] # here you get the files needed
		import os
		s7 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp7'+s7[1])
		im = Image.open('temp7'+s7[1])

		f = request.FILES['file8'] # here you get the files needed
		import os
		s8 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp8'+s8[1])
		im = Image.open('temp8'+s8[1])

		f = request.FILES['file9'] # here you get the files needed
		import os
		s9 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp9'+s9[1])
		im = Image.open('temp9'+s9[1])

		f = request.FILES['file10'] # here you get the files needed
		import os
		s10 = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp10'+s10[1])
		im = Image.open('temp10'+s10[1])

		import cv2
		import numpy as np
		image1=cv2.imread('temp1'+s[1])
		image2=cv2.imread('temp2'+s2[1])
		image3=cv2.imread('temp3'+s3[1])
		image4=cv2.imread('temp4'+s4[1])
		image5=cv2.imread('temp5'+s5[1])
		image6=cv2.imread('temp6'+s6[1])
		image7=cv2.imread('temp7'+s7[1])
		image8=cv2.imread('temp8'+s8[1])
		image9=cv2.imread('temp9'+s9[1])
		image10=cv2.imread('temp10'+s10[1])
		image1=cv2.resize(image1,(200,200))
		image2=cv2.resize(image2,(200,200))
		image3=cv2.resize(image3,(200,200))
		image4=cv2.resize(image4,(200,200))
		image5=cv2.resize(image5,(200,200))
		image6=cv2.resize(image6,(200,200))
		image7=cv2.resize(image7,(200,200))
		image8=cv2.resize(image8,(200,200))
		image9=cv2.resize(image9,(200,200))
		image10=cv2.resize(image10,(200,200))
		Horizontal1=np.hstack([image1,image2])
		Horizontal2=np.hstack([image3,image4])
		Horizontal3=np.hstack([image5,image6])
		Horizontal4=np.hstack([image7,image8])
		Horizontal5=np.hstack([image9,image10])
		Vertical_attachment=np.vstack([Horizontal1,Horizontal2,Horizontal3,Horizontal4,Horizontal5])
		cv2.imwrite('STATIC/FinalCollage10'+s[1],Vertical_attachment)
		p='FinalCollage10'+s[1]
		print("p")
		return render(request,'collageresult10.html',{'p':p})
	else:
		return render(request,'collage10.html')

def collageresult10(request):
	return render(request,'collageresult10.html')

def dotted(request):
	if request.method=='POST':
		f = request.FILES['t1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])

		from PIL import Image, ImageDraw
		import numpy as np
		im = Image.open('temp'+s[1]).convert("L")
		width, height = im.size
		max_dots = 140
		background_colour = [224, 255, 255] 
		dots_colour = (0,0,139)
		if height == max(height, width):
			downsized_image = im.resize((int(height * (max_dots / width)), max_dots))
		else:
			downsized_image = im.resize((max_dots, int(height * (max_dots / width))))
		downsized_image_width, downsized_image_height = downsized_image.size
		if height == max(height, width):
			downsized_image = im.resize((int(height * (max_dots / width)), max_dots))
		else:
			downsized_image = im.resize((max_dots, int(height * (max_dots / width))))
		downsized_image_width, downsized_image_height = downsized_image.size
		multiplier = 50
		blank_img_height = downsized_image_height * multiplier
		blank_img_width = downsized_image_width * multiplier
		padding = int(multiplier / 2)
		blank_image = np.full(
			((blank_img_height), (blank_img_width), 3), background_colour, dtype=np.uint8
		)
		pil_image = Image.fromarray(blank_image)
		draw = ImageDraw.Draw(pil_image)
		downsized_image = np.array(downsized_image)
		for y in range(0, downsized_image_height):
			for x in range(0, downsized_image_width):
				k = (x * multiplier) + padding
				m = (y * multiplier) + padding
				r = int((0.6 * multiplier) * ((255 - downsized_image[y][x]) / 255))
				leftUpPoint = (k - r, m - r)
				rightDownPoint = (k + r, m + r)
				twoPointList = [leftUpPoint, rightDownPoint]
				draw.ellipse(twoPointList, fill=dots_colour)
		pil_image.save('STATIC/dotted'+s[1])
		p='dotted'+s[1]
		print("p")
		return render(request,'dottedresult.html',{'p':p})
	else:
		return render(request,'dotted.html')

def dottedresult(request):
	return render(request,'dottedresult.html')

def banner(request):
    if request.method=='POST':
        t=request.POST.get("t1")
       
        def create_image_with_text(size, text):
            global c, inc
            c = 0
            inc = 10
            img = Image.new('RGB', (600, 200), (255-(c-9),255-(c-9),255-(c-9)))
            draw = ImageDraw.Draw(img)
            draw.text((size[0], size[1]), text, font = fnt, fill=(c,c,c))
            c += inc        
            return img
# Create the frames
        frames = []
        def roll(text):
            global c
            for i in range(len(text)+1):
                new_frame = create_image_with_text((0,0), text[:i])
                frames.append(new_frame)
            c = 0
        fnt = ImageFont.truetype("arial", 36
        	)
        all_text =t.splitlines()
        [roll(text) for text in all_text]
# Save into a GIF file that loops forever
        frames[0].save('static/banner.gif', format='GIF',
               append_images=frames[1:], save_all=True, duration=80, loop=0)
        print("Done")
        return  render (request,'bannerresult.html')
    else :
        return  render (request,'banner.html')

def bannerresult(request):
	return render(request,'bannerresult.html')

def mask(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp'+s[1])

		import cv2
		import numpy as np
		image = cv2.imread('temp'+s[1])
		mask = np.zeros(image.shape, dtype=np.uint8)
		mask = cv2.circle(mask, (260, 300), 225, (255,255,255), -1)
		result = cv2.bitwise_and(image, mask)
		result[mask==0] = 255
		cv2.imshow('image', image)
		cv2.imshow('mask', mask)
		cv2.imshow('result', result)
		cv2.imwrite('STATIC/masked'+s[1], result)
		p='masked'+s[1]
		print("p")
		return render(request,'maskresult.html',{'p':p})
	else:
		return render(request,'mask.html')

def maskresult(request):
	return render(request,'maskresult.html')

def keypoint(request):
	if request.method=='POST':
		f = request.FILES['file1'] # here you get the files needed
		import os
		s = os.path.splitext(f.name)
		print(s[1])
		handle_uploaded_file(f,'temp'+s[1])
		im = Image.open('temp'+s[1])

		import cv2
		import matplotlib.pyplot as plt
		imageread = cv2.imread('temp'+s[1])
		imagegray = cv2.cvtColor(imageread, cv2.COLOR_BGR2GRAY)
		features = cv2.SIFT_create()
		keypoints = features.detect(imagegray, None)
		output_image = cv2.drawKeypoints(imagegray, keypoints, 0, (255, 0, 0),
			                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		plt.imsave('static/keypoint'+s[1],output_image)
		p='keypoint'+s[1]
		print("p")
		return render(request,'keypointresult.html',{'p':p})
	else:
		return render(request,'keypoint.html')

def keypointresult(request):
	return render(request,'keypointresult.html')

def aboutus(request):
	return render(request,'aboutus.html')

def gif2(request):
    if request.method=='POST':
        f = request.FILES['file1']
        handle_uploaded_file(f,"g2.jpg")
        # Create instance of point object using an image
        import pointillism as pt
        point = pt.image(location='g2.jpg', debug = True)
# Crop
        point.crop([1000,500], False)
# Display original
        point.display(original=True)
        point.make('balanced')
# Display and save
        point.display()
        point.save_out(location='static', suffix='basic test')
# Apply a colormap
        point.colormap('cyanotype')
# Reset output canvas
        point._newImage(border=100)
# Make and display
        point.make('fine')
        point.display()
# Apply a colormap
        point.colormap('sepia')
# Reset output canvas
        point._newImage(border=100)
# Make and display
        point.make('fine')
        point.display()
# Create instance
        point = pt.pipeline(location='g2.jpg', debug = True, border = 0)
# Render
        point.make_gif(kind='loop', location='static', name='g2.gif', crop=True)
# Display
        from IPython.display import HTML
        HTML('<img src="g2.gif">')
        point = pt.pipeline(location='g2.jpg', debug = True, border = 0)
# Render
        point.make_gif(kind='multiplier', location='static', name='g2.gif', crop=True)
        from IPython.display import HTML
        HTML('<img src="g2.gif">')
        p='g2.gif'
        return  render (request,'displaygif2.html' ,{'p':p})
    else :
        return  render (request,'gif2.html')

def displaygif2(request):
	return render(request,'displaygif2.html')

def imagesearch(request):
	if request.method=="POST":
		t1 = request.POST.get('t1')
		print("t1",t1)
		import os
		for file in os.listdir(r'STATIC/searchimg'):
			print(file)
			os.remove('STATIC/searchimg/'+str(file)) 
		
		
		from icrawler.builtin import GoogleImageCrawler
		google_Crawler = GoogleImageCrawler(storage = {'root_dir': r'STATIC/searchimg'})
		google_Crawler.crawl(keyword = t1, max_num = 9)
		#import shutil
		#shutil.make_archive("searched_images", 'zip', r'/static/searchimg')
		l1=[]
		for file in os.listdir(r'STATIC/searchimg'):
			print(file)
			l1.append('searchimg/'+file)
			print(l1)
		return render(request,'imagesearchresult.html',{'l':l1})
	else:

		return render(request,'imagesearch.html')

def imagesearchresult(request):
	return render(request,'imagesearchresult.html')

def sidebar1(request):
	return render(request,'sidebar1.html')

def spare(request):
	return render(request,'spare.html')

def test1(request):
	return render(request,'test1.html')

def editpicture(request):
	return render(request,'editpicture.html')

def upload_file(f,name):
    destination=open("media/"+name,'wb+')
    for chunk in f.chunks():
        destination.write(chunk)
    destination.close()

def editpicture(request):
    if not request.session.has_key('userid'):
        return redirect('/userlogin')
    if request.method=='POST':
        uid=request.session['userid']
        user = models.registeredUsers.objects.get(id=uid)

        profile=request.FILES['profile']
        upload_file(profile,profile.name)
        user.profilePicture=profile.name
        user.save()
        print(profile.name)
        return render(request,'userprofile.html',{'user':user})
    else:
        return render(request,'editpicture.html')

def dashboard(request):
	if not request.session.has_key('userid'):
		return redirect('userlogin')
	uid=request.session['userid']
	user=models.registeredUsers.objects.get(id=uid)
	print(uid)
	return render(request,'dashboard.html')






    





















































    	


	
	



	
		    





	





