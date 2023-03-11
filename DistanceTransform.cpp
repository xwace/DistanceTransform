//
// Created by yc on 23-3-10.
//

#include "DistanceTransform.h"
#include <opencv2/opencv.hpp>
#include <iterator>

using namespace cv;

namespace DT{

    extern const uchar icvSaturate8u_cv[];
    #define CV_FAST_CAST_8U(t)  ( (-256 <= (t) && (t) <= 512) ? icvSaturate8u_cv[(t)+256] : 0 )
    #define CV_CALC_MIN_8U(a,b) (a) -= CV_FAST_CAST_8U((a) - (b))

    static const int DIST_SHIFT = 16;
    static const int INIT_DIST0 = INT_MAX;
    static const int DIST_MAX   = (INT_MAX >> 2);
    #define  CV_FLT_TO_FIX(x,n)  cvRound((x)*(1<<(n)))

    static void
    initTopBottom( Mat& temp, int border )
    {
        Size size = temp.size();
        for( int i = 0; i < border; i++ )
        {
            int* ttop = temp.ptr<int>(i);
            int* tbottom = temp.ptr<int>(size.height - i - 1);

            for( int j = 0; j < size.width; j++ )
            {
                ttop[j] = INIT_DIST0;
                tbottom[j] = INIT_DIST0;
            }
        }
    }

    static void
    distanceTransform_3x3( const Mat& _src, Mat& _temp, Mat& _dist, const float* metrics )
    {
        const int BORDER = 1;
        int i, j;
        const unsigned int HV_DIST = CV_FLT_TO_FIX( metrics[0], DIST_SHIFT );
        const unsigned int DIAG_DIST = CV_FLT_TO_FIX( metrics[1], DIST_SHIFT );
        const float scale = 1.f/(1 << DIST_SHIFT);

        const uchar* src = _src.ptr();
        int* temp = _temp.ptr<int>();
        float* dist = _dist.ptr<float>();
        int srcstep = (int)(_src.step/sizeof(src[0]));
        int step = (int)(_temp.step/sizeof(temp[0]));
        int dststep = (int)(_dist.step/sizeof(dist[0]));
        Size size = _src.size();

        initTopBottom( _temp, BORDER );

        // forward pass
        for( i = 0; i < size.height; i++ )
        {
            const uchar* s = src + i*srcstep;
            unsigned int* tmp = (unsigned int*)(temp + (i+BORDER)*step) + BORDER;

            //the first and last element of each row set to INIT_DIST0
            for( j = 0; j < BORDER; j++ )
                tmp[-j-1] = tmp[size.width + j] = INIT_DIST0;

            for( j = 0; j < size.width; j++ )
            {
                if( !s[j] )
                    tmp[j] = 0;
                else
                {
                    /* a0 a1 a2
                       a3 a  a4
                       a7 a6 a5
                    */

                    //t0 = a0 compares with (a1,a2,a3)->min
                    unsigned int t0 = tmp[j-step-1] + DIAG_DIST;

                    unsigned int t = tmp[j-step] + HV_DIST;
                    if( t0 > t ) t0 = t;
                    t = tmp[j-step+1] + DIAG_DIST;
                    if( t0 > t ) t0 = t;
                    t = tmp[j-1] + HV_DIST;
                    if( t0 > t ) t0 = t;

                    tmp[j] = t0;
                }
            }
        }

        std::cout<<"temp\n"<<_temp<<std::endl;

        // backward pass
        for( i = size.height - 1; i >= 0; i-- )
        {
            float* d = (float*)(dist + i*dststep);
            unsigned int* tmp = (unsigned int*)(temp + (i+BORDER)*step) + BORDER;

            //attention!!! reversing from right bottom now!
            for( j = size.width - 1; j >= 0; j-- )
            {
                unsigned int t0 = tmp[j];
                if( t0 > HV_DIST )
                {
                    /* a0 a1 a2
                       a3 a  a4
                       a7 a6 a5
                    */

                    //t0 = a compares with (a4,a5,a6,a7)
                    unsigned int t = tmp[j+step+1] + DIAG_DIST;
                    if( t0 > t ) t0 = t;
                    t = tmp[j+step] + HV_DIST;
                    if( t0 > t ) t0 = t;
                    t = tmp[j+step-1] + DIAG_DIST;
                    if( t0 > t ) t0 = t;
                    t = tmp[j+1] + HV_DIST;
                    if( t0 > t ) t0 = t;
                    tmp[j] = t0;
                }
                t0 = (t0 > DIST_MAX) ? DIST_MAX : t0;
//            d[j] = (float)(t0);
                d[j] = (float)(t0 * scale);
            }
        }
    }

    static void
    distanceATS_L1_8u( const Mat& src, Mat& dst )
    {
        int width = src.cols, height = src.rows;

        int a;
        uchar lut[256];
        int x, y;

       std::cout<<src<<std::endl;

        const uchar *sbase = src.ptr();
        uchar *dbase = dst.ptr();
        int srcstep = (int)src.step;
        int dststep = (int)dst.step;

        CV_Assert( src.type() == CV_8UC1 && dst.type() == CV_8UC1 );
        CV_Assert( src.size() == dst.size() );

        ////////////////////// forward scan ////////////////////////
        for( x = 0; x < 256; x++ )
            lut[x] = cv::saturate_cast<uchar>(x+1);

        //init first pixel to max (we're going to be skipping it)
        dbase[0] = (uchar)(sbase[0] == 0 ? 0 : 255);

        //first row (scan west only, skip first pixel)
        for( x = 1; x < width; x++ )
        {
            dbase[x] = (uchar)(sbase[x] == 0 ? 0 : lut[dbase[x-1]]);
        }

        for( y = 1; y < height; y++ )
        {
            sbase += srcstep;
            dbase += dststep;

            //for left edge, scan north only
            a = sbase[0] == 0 ? 0 : lut[dbase[-dststep]];
            dbase[0] = (uchar)a;

            for( x = 1; x < width; x++ )
            {
                a = sbase[x] == 0 ? 0 : lut[MIN(a, dbase[x - dststep])];
                dbase[x] = (uchar)a;
            }
        }

        ////////////////////// backward scan ///////////////////////

        a = dbase[width-1];

        // do last row east pixel scan here (skip bottom right pixel)
        for( x = width - 2; x >= 0; x-- )
        {
            a = lut[a];
            dbase[x] = (uchar)(CV_CALC_MIN_8U(a, dbase[x]));
        }

        // right edge is the only error case
        for( y = height - 2; y >= 0; y-- )
        {
            dbase -= dststep;

            // do right edge
            a = lut[dbase[width-1+dststep]];
            a = dbase[width-1] = (uchar)(MIN(a, dbase[width-1]));

            for( x = width - 2; x >= 0; x-- )
            {
                int b = dbase[x+dststep];
                a = lut[MIN(a, b)];
                a = MIN(a, dbase[x]);
                dbase[x] = (uchar)(a);
            }
        }
    }

    static void distanceTransform_L1_8U(InputArray _src, OutputArray _dst)
    {
        Mat src = _src.getMat();
        CV_Assert( src.type() == CV_8UC1);

        _dst.create( src.size(), CV_8UC1);
        Mat dst = _dst.getMat();

        distanceATS_L1_8u(src, dst);
    }

    struct DTColumnInvoker : ParallelLoopBody
    {
        DTColumnInvoker( const Mat* _src, Mat* _dst, const int* _sat_tab, const float* _sqr_tab)
        {
            src = _src;
            dst = _dst;
            sat_tab = _sat_tab + src->rows*2 + 1;
            sqr_tab = _sqr_tab;
        }

        void operator()(const Range& range) const CV_OVERRIDE
        {
            int i, i1 = range.start, i2 = range.end;
            int m = src->rows;
            size_t sstep = src->step, dstep = dst->step/sizeof(float);
            AutoBuffer<int> _d(m);
            int* d = _d.data();

            //deal with each colums:[0->src.cols]
            for( i = i1; i < i2; i++ )
            {
                //scan last row, sptr moves from left to the right of the row as i ascends
                const uchar* sptr = src->ptr(m-1) + i;
                float* dptr = dst->ptr<float>() + i;
                int j, dist = m-1;

                //sptr moves from last row to the first
                for( j = m-1; j >= 0; j--, sptr -= sstep )
                {
                    dist = (dist + 1) & (sptr[0] == 0 ? 0 : -1);
                    d[j] = dist;
                }

                dist = m-1;
                for( j = 0; j < m; j++, dptr += dstep )
                {
                    dist = dist + 1 - sat_tab[dist - d[j]];
                    d[j] = dist;
                    dptr[0] = sqr_tab[dist];
                }
            }
        }

        const Mat* src;
        Mat* dst;
        const int* sat_tab;
        const float* sqr_tab;
    };

    struct DTRowInvoker : ParallelLoopBody
    {
        DTRowInvoker( Mat* _dst, const float* _sqr_tab, const float* _inv_tab )
        {
            dst = _dst;
            sqr_tab = _sqr_tab;
            inv_tab = _inv_tab;
        }

        void operator()(const Range& range) const CV_OVERRIDE
        {
            const float inf = 1e15f;
            int i, i1 = range.start, i2 = range.end;
            int n = dst->cols;
            AutoBuffer<uchar> _buf((n+2)*2*sizeof(float) + (n+2)*sizeof(int));
            float* f = (float*)_buf.data();
            float* z = f + n;
            int* v = alignPtr((int*)(z + n + 1), sizeof(int));

            for( i = i1; i < i2; i++ )
            {
                float* d = dst->ptr<float>(i);
                int p, q, k;

                v[0] = 0;
                z[0] = -inf;
                z[1] = inf;
                f[0] = d[0];

                for( q = 1, k = 0; q < n; q++ )
                {
                    float fq = d[q];
                    f[q] = fq;

                    for(;;k--)
                    {
                        p = v[k];
                        float s = (fq + sqr_tab[q] - d[p] - sqr_tab[p])*inv_tab[q - p];
                        if( s > z[k] )
                        {
                            k++;
                            v[k] = q;
                            z[k] = s;
                            z[k+1] = inf;
                            break;
                        }
                    }
                }

                for( q = 0, k = 0; q < n; q++ )
                {
                    while( z[k+1] < q )
                        k++;
                    p = v[k];
                    d[q] = std::sqrt(sqr_tab[std::abs(q - p)] + f[p]);
                }
            }
        }

        Mat* dst;
        const float* sqr_tab;
        const float* inv_tab;
    };

    static void
    trueDistTrans( const Mat& src, Mat& dst )
    {
        const float inf = 1e15f;

        CV_Assert( src.size() == dst.size() );

        CV_Assert( src.type() == CV_8UC1 && dst.type() == CV_32FC1 );
        int i, m = src.rows, n = src.cols;

        cv::AutoBuffer<uchar> _buf(std::max(m*2*sizeof(float) + (m*3+1)*sizeof(int), n*2*sizeof(float)));

        // stage 1: compute 1d distance transform of each column
        float* sqr_tab = (float*)_buf.data();
        int* sat_tab = cv::alignPtr((int*)(sqr_tab + m*2), sizeof(int));
        int shift = m*2;

        for( i = 0; i < m; i++ )
            sqr_tab[i] = (float)(i*i);
        for( i = m; i < m*2; i++ )
            sqr_tab[i] = inf;
        for( i = 0; i < shift; i++ )
            sat_tab[i] = 0;
        for( ; i <= m*3; i++ )
            sat_tab[i] = i - shift;

        cv::parallel_for_(cv::Range(0, n), DTColumnInvoker(&src, &dst, sat_tab, sqr_tab), src.total()/(double)(1<<16));

        // stage 2: compute modified distance transform for each row
        float* inv_tab = sqr_tab + n;

        inv_tab[0] = sqr_tab[0] = 0.f;
        for( i = 1; i < n; i++ )
        {
            inv_tab[i] = (float)(0.5/i);
            sqr_tab[i] = (float)(i*i);
        }

        cv::parallel_for_(cv::Range(0, m), DTRowInvoker(&dst, sqr_tab, inv_tab));
    }

    static void getDistanceTransformMask( int maskType, float *metrics )
    {
        CV_Assert( metrics != 0 );

        switch (maskType)
        {
            case 30:
                metrics[0] = 1.0f;
                metrics[1] = 1.0f;
                break;

            case 31:
                metrics[0] = 1.0f;
                metrics[1] = 2.0f;
                break;

            case 32:
                metrics[0] = 0.955f;
                metrics[1] = 1.3693f;
                break;

            case 50:
                metrics[0] = 1.0f;
                metrics[1] = 1.0f;
                metrics[2] = 2.0f;
                break;

            case 51:
                metrics[0] = 1.0f;
                metrics[1] = 2.0f;
                metrics[2] = 3.0f;
                break;

            case 52:
                metrics[0] = 1.0f;
                metrics[1] = 1.4f;
                metrics[2] = 2.1969f;
                break;
            default:
                CV_Error(Error::StsBadArg, "Unknown metric type");
        }
    }

    // Wrapper function for distance transform group
    void mydistanceTransform( InputArray _src, OutputArray _dst, OutputArray _labels,
                                int distType, int maskSize, int labelType )
    {
        Mat src = _src.getMat(), labels;
        bool need_labels = _labels.needed();

        CV_Assert( src.type() == CV_8UC1);

        _dst.create( src.size(), CV_32F);
        Mat dst = _dst.getMat();

        if( need_labels )
        {
            CV_Assert( labelType == DIST_LABEL_PIXEL || labelType == DIST_LABEL_CCOMP );

            _labels.create(src.size(), CV_32S);
            labels = _labels.getMat();
            maskSize = DIST_MASK_5;
        }

        float _mask[5] = {0};

        if( maskSize != DIST_MASK_3 && maskSize != DIST_MASK_5 && maskSize != DIST_MASK_PRECISE )
            CV_Error( Error::StsBadSize, "Mask size should be 3 or 5 or 0 (precise)" );

        if( distType == DIST_C || distType == DIST_L1 )
            maskSize = !need_labels ? DIST_MASK_3 : DIST_MASK_5;
        else if( distType == DIST_L2 && need_labels )
            maskSize = DIST_MASK_5;

        if( maskSize == DIST_MASK_PRECISE )
        {
            trueDistTrans( src, dst );
            return;
        }

        CV_Assert( distType == DIST_C || distType == DIST_L1 || distType == DIST_L2 );

        getDistanceTransformMask( (distType == DIST_C ? 0 :
                                   distType == DIST_L1 ? 1 : 2) + maskSize*10, _mask );

        Size size = src.size();

        int border = maskSize == DIST_MASK_3 ? 1 : 2;
        Mat temp( size.height + border*2, size.width + border*2, CV_32SC1 );

        if( !need_labels )
        {
            if( maskSize == DIST_MASK_3 )
            {
                distanceTransform_3x3(src, temp, dst, _mask);
            }
            else
            {
//                distanceTransform_5x5(src, temp, dst, _mask);
            }
        }
        else
        {
            labels.setTo(Scalar::all(0));

            if( labelType == DIST_LABEL_CCOMP )
            {
                Mat zpix = src == 0;
                connectedComponents(zpix, labels, 8, CV_32S, CCL_WU);
            }
            else
            {
                int k = 1;
                for( int i = 0; i < src.rows; i++ )
                {
                    const uchar* srcptr = src.ptr(i);
                    int* labelptr = labels.ptr<int>(i);

                    for( int j = 0; j < src.cols; j++ )
                        if( srcptr[j] == 0 )
                            labelptr[j] = k++;
                }
            }

//            distanceTransformEx_5x5( src, temp, dst, labels, _mask );
        }
    }

    void distanceTransform( InputArray _src, OutputArray _dst,
                                int distanceType, int maskSize, int dstType)
    {
        if (distanceType == DIST_L1 && dstType==CV_8U)
            distanceTransform_L1_8U(_src, _dst);
        else
            mydistanceTransform(_src, _dst, noArray(), distanceType, maskSize, DIST_LABEL_PIXEL);

    }

    const uchar icvSaturate8u_cv[] =
    {
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
            16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
            32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
            48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,
            64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
            80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
            96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
            112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
            128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
            144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
            160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
            176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
            192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
            208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
            224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
            240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255
    };
}
