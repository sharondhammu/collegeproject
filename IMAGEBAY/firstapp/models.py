from django.db import models
from datetime import date
# Create your models here.

class enquiry(models.Model):
	username=models.CharField(max_length=40)
	emailid=models.EmailField()
	message=models.TextField()

class registeredUsers(models.Model):
	username=models.CharField(max_length=40)
	emailid=models.EmailField()
	password=models.CharField(max_length=40)
	registerDate=models.DateField(default=date.today())
	phoneNumber=models.CharField(max_length=10,null=True)
	profilePicture=models.ImageField(null=True,blank=True)
	confirmpassword=models.CharField(max_length=100)
	dob=models.CharField(null=True, blank=True, max_length=100)
	gender=models.CharField(max_length=1,choices=[('m','male'),('f','female')],default='f')
	city=models.CharField(max_length=100, null=True, blank=True)
	state=models.CharField(max_length=100, null=True, blank=True)
	def __str__(self):
	   return self.username

class reviews(models.Model):
	subject=models.CharField(max_length=500)
	message=models.TextField()
	user=models.ForeignKey(registeredUsers,on_delete=models.CASCADE)
	reviewDate=models.DateField(default=date.today())

class blogs(models.Model):
	title=models.CharField(max_length=200)
	image=models.ImageField()
	content=models.TextField()
	registerDate=models.DateField(default=date.today())

class tools(models.Model):
	toolname=models.TextField()
	image=models.ImageField()
	description=models.TextField()

class guide(models.Model):	
    tool=models.ForeignKey(tools,on_delete=models.CASCADE)
    step=models.TextField()
    image=models.ImageField()

class glossary(models.Model):
    word=models.CharField(max_length=500)
    meaning=models.TextField()

class tutorials(models.Model):
    title=models.CharField(max_length=200)
    video=models.FileField()
    uploadingDate=models.DateField(default=date.today())    




	




