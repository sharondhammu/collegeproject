# Generated by Django 4.0.3 on 2022-03-28 19:44

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('firstapp', '0005_alter_blogs_registerdate_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='glossary',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('word', models.CharField(max_length=500)),
                ('meaning', models.TextField()),
            ],
        ),
        migrations.AlterField(
            model_name='blogs',
            name='registerDate',
            field=models.DateField(default=datetime.date(2022, 3, 28)),
        ),
        migrations.AlterField(
            model_name='registeredusers',
            name='registerDate',
            field=models.DateField(default=datetime.date(2022, 3, 28)),
        ),
        migrations.AlterField(
            model_name='reviews',
            name='reviewDate',
            field=models.DateField(default=datetime.date(2022, 3, 28)),
        ),
    ]
