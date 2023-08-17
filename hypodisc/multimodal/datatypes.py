#! /usr/bin/env python

from rdf.namespaces import XSD


XSD_DATEFRAG = {XSD + 'gDay',
                XSD + 'gMonth',
                XSD + 'gMonthDay'}

XSD_DATETIME = {XSD + 'date',
                XSD + 'dateTime',
                XSD + 'dateTimeStamp',
                XSD + 'gYear',
                XSD + 'gYearMonth'}

XSD_NUMERIC = {XSD + 'decimal',
               XSD + 'double',
               XSD + 'float',
               XSD + 'long',
               XSD + 'int',
               XSD + 'short',
               XSD + 'byte',
               XSD + 'integer',
               XSD + 'nonNegativeInteger',
               XSD + 'nonPositiveInteger',
               XSD + 'negativeInteger',
               XSD + 'positiveInteger',
               XSD + 'unsignedLong',
               XSD + 'unsignedInt',
               XSD + 'unsignedShort',
               XSD + 'unsignedByte'}

XSD_STRING = {XSD + 'string',
              XSD + 'normalizedString',
              XSD + 'token',
              XSD + 'language',
              XSD + 'Name',
              XSD + 'NCName',
              XSD + 'ENTITY',
              XSD + 'ID',
              XSD + 'IDREF',
              XSD + 'NMTOKEN',
              XSD + 'anyURI'}
