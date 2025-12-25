"""
PDF Parser Module

This module contains different parsers for extracting information from PDF files.
Each parser is specialized for a specific type of document.
"""

from .technical_drawing_parser import TechnicalDrawingParser

__all__ = ['TechnicalDrawingParser']
