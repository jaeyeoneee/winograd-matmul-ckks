import pytest

from src.WinogradTransform import WinogradTransform

def test_Winograd_2_3():
  m, r = 2, 3
  winograd_2_3 = WinogradTransform(m, r)
  
  assert winograd_2_3.verify_correctness()
  

def test_Winograd_4_3():
  m, r = 4, 3
  winograd_4_3 = WinogradTransform(m, r)
  
  assert winograd_4_3.verify_correctness()
  
def test_winograd_6_3():
  pass