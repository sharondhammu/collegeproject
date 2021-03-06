# Generated by Django 4.0.3 on 2022-04-02 21:15

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('firstapp', '0006_glossary_alter_blogs_registerdate_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='tutorials',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=200)),
                ('video', models.FileField(upload_to='')),
                ('uploadingDate', models.DateField(default=datetime.date(2022, 4, 2))),
            ],
        ),
        migrations.AlterField(
            model_name='blogs',
            name='registerDate',
            field=models.DateField(default=datetime.date(2022, 4, 2)),
        ),
        migrations.AlterField(
            model_name='registeredusers',
            name='registerDate',
            field=models.DateField(default=datetime.date(2022, 4, 2)),
        ),
        migrations.AlterField(
            model_name='reviews',
            name='reviewDate',
            field=models.DateField(default=datetime.date(2022, 4, 2)),
        ),
    ]
