from django.contrib import admin

from firstapp import models

# Register your models here.

admin.site.register(models.enquiry)

admin.site.register(models.registeredUsers)

admin.site.register(models.reviews)
admin.site.register(models.tools)
admin.site.register(models.guide)
admin.site.register(models.blogs)
admin.site.register(models.tutorials)
admin.site.register(models.glossary)





