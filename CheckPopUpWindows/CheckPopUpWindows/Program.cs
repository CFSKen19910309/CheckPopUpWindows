using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System.Drawing;
using Emgu.CV.Util;

namespace CheckPopUpWindows
{
    class Program
    {
        static void Main(string[] args)
        {
            DateTime t_StartTime = DateTime.Now;
            string[] TruePopUpWindows = System.IO.Directory.GetFiles(@"C:\Users\FD_Kenzo_NB\Desktop\PopUpWindows\True");
            string[] FalsePopUpWindows = System.IO.Directory.GetFiles(@"C:\Users\FD_Kenzo_NB\Desktop\PopUpWindows\False");
            foreach (string t_FileName in TruePopUpWindows)
            {
                System.Console.WriteLine("File Name :{0}", t_FileName);
                Emgu.CV.Mat t_Mat = new Emgu.CV.Mat(t_FileName);
                DoCheckPopUp_v1(t_Mat);
                t_Mat.Dispose();
            }
            foreach (string t_FileName in FalsePopUpWindows)
            {
                System.Console.WriteLine("File Name :{0}", t_FileName);
                Emgu.CV.Mat t_Mat = new Emgu.CV.Mat(t_FileName);
                DoCheckPopUp_v1(t_Mat);
                t_Mat.Dispose();
            }
            DateTime t_EndTime = DateTime.Now;
            TimeSpan t_SpentTime = t_EndTime - t_StartTime;
            System.Console.WriteLine("Spent Time :{0} ms", t_SpentTime.TotalMilliseconds);

        }
        public static Tuple<bool, Rectangle> DoCheckPopUp_v1(Emgu.CV.Mat f_Source)
        {
            Emgu.CV.CvInvoke.NamedWindow("Original", Emgu.CV.CvEnum.NamedWindowType.Normal);
            Emgu.CV.CvInvoke.NamedWindow("Test", Emgu.CV.CvEnum.NamedWindowType.Normal);
            Emgu.CV.CvInvoke.NamedWindow("B", Emgu.CV.CvEnum.NamedWindowType.Normal);
            Emgu.CV.CvInvoke.NamedWindow("G", Emgu.CV.CvEnum.NamedWindowType.Normal);
            Emgu.CV.CvInvoke.NamedWindow("R", Emgu.CV.CvEnum.NamedWindowType.Normal);
            Emgu.CV.CvInvoke.NamedWindow("t_SplitMatThreshold_B", Emgu.CV.CvEnum.NamedWindowType.Normal);
            Emgu.CV.CvInvoke.NamedWindow("t_SplitMatThreshold_G", Emgu.CV.CvEnum.NamedWindowType.Normal);
            Emgu.CV.CvInvoke.NamedWindow("t_SplitMatThreshold_R", Emgu.CV.CvEnum.NamedWindowType.Normal);
            Emgu.CV.CvInvoke.NamedWindow("Merge", Emgu.CV.CvEnum.NamedWindowType.Normal);

            bool ret = false;
            Rectangle retR = Rectangle.Empty;
            Emgu.CV.Mat[] t_SplitMat = f_Source.Split();
            Emgu.CV.Mat[] t_SplitMatThreshold = new Emgu.CV.Mat[3];
            Emgu.CV.Mat t_MergeThrshold = new Emgu.CV.Mat(f_Source.Rows, f_Source.Cols, Emgu.CV.CvEnum.DepthType.Cv8U, 1);
            t_MergeThrshold.SetTo(new Emgu.CV.Structure.MCvScalar(255, 255, 255));
            for (int i = 0; i < f_Source.NumberOfChannels; i ++ )
            {
                t_SplitMatThreshold[i] = new Emgu.CV.Mat();
                Emgu.CV.CvInvoke.Threshold(t_SplitMat[i], t_SplitMatThreshold[i], 200, 255, Emgu.CV.CvEnum.ThresholdType.Binary);
                Emgu.CV.CvInvoke.BitwiseAnd(t_MergeThrshold, t_SplitMatThreshold[i], t_MergeThrshold);
                //Emgu.CV.CvInvoke.Imshow("Merge", t_MergeThrshold);
                //Emgu.CV.CvInvoke.WaitKey(0);

            }

            Emgu.CV.CvInvoke.Erode(t_MergeThrshold, t_MergeThrshold, null, new Point(-1, -1), 3, Emgu.CV.CvEnum.BorderType.Default, new Emgu.CV.Structure.MCvScalar(0));
            Emgu.CV.CvInvoke.Dilate(t_MergeThrshold, t_MergeThrshold, null, new Point(-1, -1), 35, Emgu.CV.CvEnum.BorderType.Default, new Emgu.CV.Structure.MCvScalar(255));
            Emgu.CV.CvInvoke.Erode(t_MergeThrshold, t_MergeThrshold, null, new Point(-1, -1), 20, Emgu.CV.CvEnum.BorderType.Default, new Emgu.CV.Structure.MCvScalar(0));
            
            //Emgu.CV.CvInvoke.Imshow("Merge", t_MergeThrshold);
            //Emgu.CV.CvInvoke.WaitKey(0);
            
            //Emgu.CV.CvInvoke.Imshow("Merge", t_MergeThrshold);
            //Emgu.CV.CvInvoke.WaitKey(0);

            Emgu.CV.Util.VectorOfVectorOfPoint t_Contours = new Emgu.CV.Util.VectorOfVectorOfPoint();
            Emgu.CV.CvInvoke.FindContours(t_MergeThrshold, t_Contours, null, Emgu.CV.CvEnum.RetrType.External, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple);
            int t_Count = t_Contours.Size;

            Emgu.CV.CvInvoke.Imshow("Original", f_Source);
            Emgu.CV.CvInvoke.Imshow("B", t_SplitMat[0]);
            Emgu.CV.CvInvoke.Imshow("G", t_SplitMat[1]);
            Emgu.CV.CvInvoke.Imshow("R", t_SplitMat[2]);
            Emgu.CV.CvInvoke.Imshow("t_SplitMatThreshold_B", t_SplitMatThreshold[0]);
            Emgu.CV.CvInvoke.Imshow("t_SplitMatThreshold_G", t_SplitMatThreshold[1]);
            Emgu.CV.CvInvoke.Imshow("t_SplitMatThreshold_R", t_SplitMatThreshold[2]);
            Emgu.CV.CvInvoke.Imshow("Merge", t_MergeThrshold);

            Emgu.CV.CvInvoke.WaitKey(5);

            Rectangle r = Rectangle.Empty;
            if(t_Count != 1)
            {
                ret = false;
                for (int i = 0; i < f_Source.NumberOfChannels; i++)
                {
                    t_SplitMat[i].Dispose();
                    t_SplitMatThreshold[i].Dispose();
                }
                t_MergeThrshold.Dispose();
                t_Contours.Clear();
                t_Contours.Dispose();
                return new Tuple<bool, Rectangle>(ret, retR);
            }
            int tt = 0;

            if( tt == 99 )
            {
                Emgu.CV.CvInvoke.WaitKey();
            }
            for (int i = 0; i < t_Count; i++)
            {
                double t_AreaCount = Emgu.CV.CvInvoke.ContourArea(t_Contours[i]);
                if (t_AreaCount > 50000)
                {
                    Emgu.CV.Util.VectorOfPoint t_ApproxContours = new VectorOfPoint();
                    Emgu.CV.Structure.RotatedRect t_RotatedRect = Emgu.CV.CvInvoke.MinAreaRect(t_Contours[i]);
                    Rectangle t_Rectangle = t_RotatedRect.MinAreaRect();
                    if (t_Rectangle.Left > 40 && t_Rectangle.Left < 90 && t_Rectangle.Right > t_MergeThrshold.Width - 90 && t_Rectangle.Right < t_MergeThrshold.Width - 40)
                    {
                        if (t_Rectangle.Height < 600 && t_Rectangle.Height > 200)
                        {
                            //if (r.IsEmpty) r = t_RotatedRect.MinAreaRect();
                            //else r = Rectangle.Union(r, t_RotatedRect.MinAreaRect());
                            r = t_Rectangle;
                            Emgu.CV.CvInvoke.DrawContours(f_Source, t_Contours, i, new Emgu.CV.Structure.MCvScalar(255, 0, 255), 10);
                        }
                    }
                    t_ApproxContours.Clear();
                    t_ApproxContours.Dispose();
                }
            }
          
            if (r.IsEmpty)
            {
                Emgu.CV.CvInvoke.Imshow("Test", f_Source);
                Emgu.CV.CvInvoke.WaitKey(30);
                for (int i = 0; i < t_Contours.Size; i++)
                {
                    Emgu.CV.CvInvoke.DrawContours(f_Source, t_Contours, i, new Emgu.CV.Structure.MCvScalar(255, 0, 255),10);
                    Emgu.CV.CvInvoke.Imshow("Test", f_Source);
                    Emgu.CV.CvInvoke.WaitKey(10);

                }
            }
            if (!r.IsEmpty)
            {
                Emgu.CV.CvInvoke.Imshow("Test", f_Source);
                ret = true;
                retR = r;
                for (int i = 0; i < t_Contours.Size; i++)
                {
                    Emgu.CV.CvInvoke.DrawContours(f_Source, t_Contours, i, new Emgu.CV.Structure.MCvScalar(255, 0, 255), 10);
                    Emgu.CV.CvInvoke.Imshow("Test", f_Source);
                    Emgu.CV.CvInvoke.WaitKey(10);
                }
            }




            for (int i = 0; i < f_Source.NumberOfChannels; i++)
            {
                t_SplitMat[i].Dispose();
                t_SplitMatThreshold[i].Dispose();
            }
            t_MergeThrshold.Dispose();
            t_Contours.Clear();
            t_Contours.Dispose();
            return new Tuple<bool, Rectangle>(ret, retR);

        }
        public static Tuple<bool, Rectangle> DoCheckPopUp_v2(Emgu.CV.Mat f_Source)
        {
            bool ret = false;
            Rectangle retR = Rectangle.Empty;
            Emgu.CV.Mat t_ProcessedImage = new Emgu.CV.Mat();
            Emgu.CV.CvInvoke.CvtColor(f_Source, t_ProcessedImage, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);
            Size t_ImageSize = t_ProcessedImage.Size;
            Emgu.CV.CvInvoke.Threshold(t_ProcessedImage, t_ProcessedImage, 230, 255, Emgu.CV.CvEnum.ThresholdType.Binary);
            Emgu.CV.CvInvoke.Dilate(t_ProcessedImage, t_ProcessedImage, null, new Point(-1, -1), 20, Emgu.CV.CvEnum.BorderType.Default, new Emgu.CV.Structure.MCvScalar(255));
            Emgu.CV.CvInvoke.Erode(t_ProcessedImage, t_ProcessedImage, null, new Point(-1, -1), 20, Emgu.CV.CvEnum.BorderType.Default, new Emgu.CV.Structure.MCvScalar(0));
            Emgu.CV.Util.VectorOfVectorOfPoint t_Contours = new Emgu.CV.Util.VectorOfVectorOfPoint();
            Emgu.CV.CvInvoke.FindContours(t_ProcessedImage, t_Contours, null, Emgu.CV.CvEnum.RetrType.External, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple);
            int t_Count = t_Contours.Size;
            if (t_Count == 1)
            {
                Emgu.CV.Structure.RotatedRect t_RotatedRect = Emgu.CV.CvInvoke.MinAreaRect(t_Contours[0]);
                //Center Of Pop Window
                if (Math.Abs(t_RotatedRect.Center.X - t_ImageSize.Width * 0.5) <= 10 && Math.Abs(t_RotatedRect.Center.Y - t_ImageSize.Height * 0.5) <= 10)
                {
                    //Area Size Setting
                    double t_AreaCount = Emgu.CV.CvInvoke.ContourArea(t_Contours[0]);
                    //if (t_AreaCount < 120000)
                    if (t_AreaCount / (t_ImageSize.Width * t_ImageSize.Height) < 0.5)
                    {
                        ret = true;
                        retR = t_RotatedRect.MinAreaRect();
                    }
                }
            }
            return new Tuple<bool, Rectangle>(ret, retR);
        }
    }

}
