' Executes a ScanProfiler SolidWorks macro against a recording mock of the SolidWorks API.
' Usage: cscript //nologo sw_mock_harness.vbs <macro.vbs>
Option Explicit

Function N(x)
    N = Replace(CStr(x), ",", ".")   ' locale-proof number output
End Function

Class MockSeg
    Public Property Let ConstructionGeometry(v)
        WScript.Echo "Construction " & v
    End Property
End Class

Class MockFeature
    Public TypeName2, NextF, FName
    Public Property Get GetTypeName2()
        GetTypeName2 = TypeName2
    End Property
    Public Function GetNextFeature()
        Set GetNextFeature = NextF
    End Function
    Public Function Select2(append, mark)
        Select2 = True
    End Function
    Public Property Let Name(v)
        FName = v
        WScript.Echo "Rename " & v
    End Property
End Class

Class MockFeatMgr
    Public Part
    Public Function InsertRefPlane(c1, d1, c2, d2, c3, d3)
        WScript.Echo "InsertRefPlane " & c1 & " " & N(d1)
        Set InsertRefPlane = Part.AddFeature("RefPlane")
    End Function
End Class

Class MockSketchMgr
    Public Part, InSketch, AddToDBv, Displayv
    Public Sub InsertSketch(flag)
        InSketch = Not InSketch
        WScript.Echo "InsertSketch " & InSketch
        If Not InSketch Then Part.AddFeature "ProfileFeature"
    End Sub
    Public Property Let AddToDB(v)
        AddToDBv = v
    End Property
    Public Property Let DisplayWhenAdded(v)
        Displayv = v
    End Property
    Private Sub NeedSketch()
        If Not InSketch Then WScript.Echo "ERROR entity outside sketch": WScript.Quit 3
    End Sub
    Public Function CreateLine(x1, y1, z1, x2, y2, z2)
        NeedSketch
        WScript.Echo "CreateLine " & N(x1) & " " & N(y1) & " " & N(z1) & " " & N(x2) & " " & N(y2) & " " & N(z2)
        Set CreateLine = New MockSeg
    End Function
    Public Function CreateArc(xc, yc, zc, x1, y1, z1, x2, y2, z2, d)
        NeedSketch
        WScript.Echo "CreateArc " & N(xc) & " " & N(yc) & " " & N(x1) & " " & N(y1) & " " & N(x2) & " " & N(y2) & " " & d
        Set CreateArc = New MockSeg
    End Function
    Public Function CreateCircleByRadius(xc, yc, zc, r)
        NeedSketch
        WScript.Echo "CreateCircleByRadius " & N(xc) & " " & N(yc) & " " & N(r)
        Set CreateCircleByRadius = New MockSeg
    End Function
    Public Function CreateSpline2(pts, natural)
        NeedSketch
        WScript.Echo "CreateSpline2 " & (UBound(pts) + 1) & " " & N(pts(0)) & " " & N(pts(1))
        Set CreateSpline2 = New MockSeg
    End Function
End Class

Class MockPart
    Public FM, SM, First, Last
    Private Sub Class_Initialize()
        Set FM = New MockFeatMgr: Set FM.Part = Me
        Set SM = New MockSketchMgr: Set SM.Part = Me: SM.InSketch = False
        Set First = Nothing: Set Last = Nothing
        AddFeature "OriginProfileFeature"
        AddFeature "RefPlane": AddFeature "RefPlane": AddFeature "RefPlane"
        AddFeature "OriginProfileFeature"
    End Sub
    Public Function AddFeature(t)
        Dim f
        Set f = New MockFeature
        f.TypeName2 = t
        Set f.NextF = Nothing
        If First Is Nothing Then Set First = f Else Set Last.NextF = f
        Set Last = f
        Set AddFeature = f
    End Function
    Public Property Get GetType()
        GetType = 1
    End Property
    Public Property Get FirstFeature()
        Set FirstFeature = First
    End Property
    Public Property Get FeatureManager()
        Set FeatureManager = FM
    End Property
    Public Property Get SketchManager()
        Set SketchManager = SM
    End Property
    Public Function FeatureByPositionReverse(i)
        Set FeatureByPositionReverse = Last
    End Function
    Public Sub ClearSelection2(all)
    End Sub
    Public Function EditRebuild3()
        EditRebuild3 = True
    End Function
    Public Sub ViewZoomtofit2()
    End Sub
End Class

Class MockSw
    Public Doc
    Private Sub Class_Initialize()
        Set Doc = New MockPart
    End Sub
    Public Property Get ActiveDoc()
        Set ActiveDoc = Doc
    End Property
End Class

Function NewMockSw()
    Set NewMockSw = New MockSw
End Function

Dim fso, src
Set fso = CreateObject("Scripting.FileSystemObject")
src = fso.OpenTextFile(WScript.Arguments(0)).ReadAll
src = Replace(src, "Option Explicit", "")
src = Replace(src, "GetObject(, ""SldWorks.Application"")", "NewMockSw()")
src = Replace(src, "CreateObject(""SldWorks.Application"")", "NewMockSw()")
src = Replace(src, "MsgBox msg, 64, ""ScanProfiler""", "WScript.Echo ""NOTIFY "" & msg")
ExecuteGlobal src
