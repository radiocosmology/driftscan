#!/bin/bash

# Zip up test products
tar czf drift_testproducts.tar.gz saved_products
scp drift_testproducts.tar.gz jrs65@prawn.cita.utoronto.ca:public_html/