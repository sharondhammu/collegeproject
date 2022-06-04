"""IMAGEBAY URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from firstapp import views

from django.conf import settings
from django.contrib.staticfiles.urls import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.urls import re_path as url

urlpatterns = [
    path('admin/', admin.site.urls),
    path('mytest',views.test),
    path('contact',views.contact,name="contactus"),
    path('allblogs',views.allblogs,name="allblogs"),
    path('forgotpassword',views.forgotpassword, name="forgotpassword"),
    path('userlogin',views.userlogin,name="userlogin"),
    path('userregister',views.userregister,name="userregister"),
    path('index',views.index,name="index"),
    path('home',views.home,name="home"),
    path('readfullblog/<int:id>',views.readfullblog,name="readfullblog"),
    path('userprofile',views.userprofile),
    path('addreviews',views.addreviews, name="reviews"),
    path('myprofile',views.myprofile,name="myprofile"),
    path('changepassword',views.changepassword,name="changepassword"),
    path('glossary',views.glossary,name="glossary"),
    path('glossary/<str:id>',views.glossaryletter,name="glossaryletter"),
    path('logout',views.logout,name="logout"),
    path('tutorial',views.tutorial,name="tutorial"),
    path('Tools',views.Tools,name="tools"),
    url(r'^search_blog$',views.search_blog,name="search_blog"),
    path('news',views.news,name="livenews"),
    path('cropimage',views.cropimage,name="cropanimage"),
    path('resultpage',views.resultpage,name="resultingpage"),
    path('transposeimage',views.transposeimage,name="transposeanimage"),
    path('forgotpasswordemailsent',views.forgotpasswordemailsent,name=""),
    path('blureffect',views.blureffect,name="blureffect"),
    path('colorswap',views.colorswap,name="colorswap"),
    path('resize',views.resize,name="resize"),
    path('grayscale',views.grayscale,name="grayscale"),
    path('facerecognition',views.facerecognition,name="facerecognition"),
    path('facerecognitionresult',views.facerecognitionresult,name="facerecognitionresult"), 
    path('addlogo',views.addlogo,name="addlogo"),
    path('splitmerge',views.splitmerge,name="splitmerge"),
    path('imagevideo',views.imagevideo,name="imagevideo"),
    path('imagevideoresult',views.imagevideoresult,name="imagevideoresult"),
    path('watermark',views.watermark,name="watermark"),
    path('watermarkresult',views.watermarkresult,name="watermarkresult"),
    path('cartoonimage',views.cartoonimage,name="cartoonimage"),
    path('filters',views.filters,name="filters"),
    path('blend',views.blend,name="blend"),
    path('blendresult',views.blendresult,name="blendresult"),
    path('analysis',views.analysis,name="analysis"),
    path('textimage',views.textimage,name="textimage"),
    path('textimageresult',views.textimageresult,name="textimageresult"),
    path('textextract',views.textextract,name="textextract"),
    path('classification',views.classification,name="classification"),
    path('detection',views.detection,name="detection"),
    path('gif',views.gif,name="gif"),
    path('gifresult',views.gifresult,name="gifresult"),
    path('transposeresult',views.transposeresult,name="transposeresult"),
    path('blurresult',views.blurresult,name="blurresult"),
    path('cartoonresult',views.cartoonresult,name="cartoonresult"),
    path('resizeresult',views.resizeresult,name="resizeresult"),
    path('grayscaleresult',views.grayscaleresult,name="grayscaleresult"),
    path('contour',views.contour,name="contour"),
    path('contourresult',views.contourresult,name="contourresult"),
    path('edgeenhance',views.edgeenhance,name="edgeenhance"),
    path('edgeenhanceresult',views.edgeenhanceresult,name="edgeenhanceresult"),
    path('emboss',views.emboss,name="emboss"),
    path('embosseffect',views.embosseffect,name="embosseffect"),
    path('findedge',views.findedge,name="findedge"),
    path('smooth',views.smooth,name="smooth"),
    path('smoothresult',views.smoothresult,name="smoothresult"),
    path('sharpen',views.sharpen,name="sharpen"),
    path('sharpenresult',views.sharpenresult,name="sharpenresult"),
    path('sepia',views.sepia,name="sepia"),
    path('sepiaresult',views.sepiaresult,name="sepiaresult"),
    path('pencil',views.pencil,name="pencil"),
    path('pencilresult',views.pencilresult,name="pencilresult"),
    path('HDR',views.HDR,name="HDR"),
    path('HDRresult',views.HDRresult,name="HDRresult"),
    path('invert',views.invert,name="invert"),
    path('invertresult',views.invertresult,name="invertresult"),
    path('summer',views.summer,name="summer"),
    path('summerresult',views.summerresult,name="summerresult"),
    path('winter',views.winter,name="winter"),
    path('winterresult',views.winterresult,name="winterresult"),
    path('pngjpg',views.pngjpg,name="pngjpg"),
    path('pngjpgresult',views.pngjpgresult,name="pngjpgresult"),
    path('imagetext',views.imagetext,name="imagetext"),
    path('movingtext',views.movingtext,name="movingtext"),
    path('movingtextresult',views.movingtextresult,name="movingtextresult"),
    path('drawellipse',views.drawellipse,name="drawellipse"),
    path('drawellipseresult',views.drawellipseresult,name="drawellipseresult"),
    path('brightness',views.brightness,name="brightness"),
    path('brightnessresult',views.brightnessresult,name="brightnessresult"),
    path('contrast',views.contrast,name="contrast"),
    path('contrastresult',views.contrastresult,name="contrastresult"),
    path('saturation',views.saturation,name="saturation"),
    path('saturationresult',views.saturationresult,name="saturationresult"),
    path('vignette',views.vignette,name="vignette"),
    path('vignetteresult',views.vignetteresult,name="vignetteresult"),
    path('adjust',views.adjust,name="adjust"),
    path('adjustresult',views.adjustresult,name="adjustresult"),
    path('sincity',views.sincity,name="sincity"),
    path('sincityresult',views.sincityresult,name="sincityresult"),
    path('vintage',views.vintage,name="vintage"),
    path('vintageresult',views.vintageresult,name="vintageresult"),
    path('collage',views.collage,name="collage"),
    path('collageresult',views.collageresult,name="collageresult"),
    path('collage2',views.collage2,name="collage2"),
    path('collage2result',views.collage2result,name="collage2result"),
    path('collage6',views.collage6,name="collage6"),
    path('collageresult6',views.collageresult6,name="collageresult6"),
    path('collage8',views.collage8,name="collage8"),
    path('collageresult8',views.collageresult8,name="collageresult8"),
    path('collage10',views.collage10,name="collage10"),
    path('collageresult10',views.collageresult10,name="collageresult10"),
    path('dotted',views.dotted,name="dotted"),
    path('dottedresult',views.dottedresult,name="dottedresult"),
    path('banner',views.banner,name="banner"),
    path('bannerresult',views.bannerresult,name="bannerresult"),
    path('mask',views.mask,name="mask"),
    path('maskresult',views.maskresult,name="maskresult"),
    path('keypoint',views.keypoint,name="keypoint"),
    path('keypointresult',views.keypointresult,name="keypointresult"),
    path('aboutus',views.aboutus,name="aboutus"),
    path('editprofile',views.editprofile,name="editprofile"),
    path('userprofile',views.userprofile,name="userprofile"),
    path('gif2',views.gif2,name="gif2"),
    path('displaygif2',views.displaygif2,name="displaygif2"),
    path('imagesearch',views.imagesearch,name="imagesearch"),
    path('imagesearchresult',views.imagesearchresult,name="imagesearchresult"),
    path('sidebar1',views.sidebar1,name="sidebar1"),
    path('spare',views.spare,name="spare"),
    path('test1',views.test1,name="test1"),
    path('editpicture',views.editpicture,name="editpicture"),
    path('dashboard',views.dashboard,name="dashboard"),




















   
   
   
   
   
   









]

urlpatterns+=staticfiles_urlpatterns()
urlpatterns+=static(settings.MEDIA_URL, document_root=settings.MEDIA_DIR)
