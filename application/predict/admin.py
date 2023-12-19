from django.contrib import admin
from django.http.request import HttpRequest
from predict.models import RegressionModel


@admin.register(RegressionModel)
class RegressionModelAdmin(admin.ModelAdmin):
    list_display = ( 'training_file', 'regression_file', 'legacy_date')
    list_filter = ( 'training_file', 'regression_file', 'legacy_date')
    search_fields = ( 'training_file', 'regression_file', 'legacy_date')
    
    def has_add_permission(self, request: HttpRequest) -> bool:
        return False
    
    def has_delete_permission(self, request: HttpRequest) -> bool:
        return False
    
    def has_change_permission(self, request: HttpRequest) -> bool:
        return False
    
