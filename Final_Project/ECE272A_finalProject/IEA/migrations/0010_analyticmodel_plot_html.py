# Generated by Django 3.2.9 on 2021-11-27 02:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('IEA', '0009_remove_analyticmodel_analytic_plot'),
    ]

    operations = [
        migrations.AddField(
            model_name='analyticmodel',
            name='plot_html',
            field=models.TextField(default=''),
        ),
    ]
