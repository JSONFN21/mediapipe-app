package com.google.mediapipe.examples.facelandmarker

/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import kotlin.math.max
import kotlin.math.min

class OverlayView(context: Context?, attrs: AttributeSet?) :
    View(context, attrs) {

    private var results: FaceLandmarkerResult? = null
    private var linePaint = Paint()
    private var pointPaint = Paint()
    private var guidePaint = Paint()
    private var boxPaint = Paint()
    private var focusPaint = Paint()

    private var scaleFactor: Float = 1f
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

    var showCenteringGuide = false
    var showEyeAlignmentBox = false
    var eyeAlignmentBox: RectF? = null
    var frozenEyeBox: RectF? = null
    var focusRing: RectF? = null
    var isTargetingRightEye: Boolean? = null // True = Right Eye, False = Left Eye, Null = None/Reset
    
    // Set to false by default to hide face mesh as requested
    var showLandmarks = false

    init {
        initPaints()
    }

    fun clear() {
        results = null
        linePaint.reset()
        pointPaint.reset()
        initPaints()
        invalidate()
    }

    private fun initPaints() {
        linePaint.color =
            ContextCompat.getColor(context!!, R.color.mp_color_primary)
        linePaint.strokeWidth = LANDMARK_STROKE_WIDTH
        linePaint.style = Paint.Style.STROKE

        pointPaint.color = Color.YELLOW
        pointPaint.strokeWidth = LANDMARK_STROKE_WIDTH
        pointPaint.style = Paint.Style.FILL

        guidePaint.color = Color.GREEN
        guidePaint.strokeWidth = 5f
        guidePaint.style = Paint.Style.STROKE
        
        boxPaint.color = Color.CYAN
        boxPaint.strokeWidth = 5f
        boxPaint.style = Paint.Style.STROKE

        focusPaint.color = Color.GREEN
        focusPaint.strokeWidth = 6f
        focusPaint.style = Paint.Style.STROKE
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        if (showCenteringGuide) {
            val cx = width / 2f
            val cy = height / 2f
            val len = 100f
            canvas.drawLine(cx - len, cy, cx + len, cy, guidePaint)
            canvas.drawLine(cx, cy - len, cx, cy + len, guidePaint)
        }

        // Calculate scaling and offsets
        val scaledImageWidth = imageWidth * scaleFactor
        val scaledImageHeight = imageHeight * scaleFactor
        val offsetX = (width - scaledImageWidth) / 2f
        val offsetY = (height - scaledImageHeight) / 2f

        val boxToDraw = frozenEyeBox ?: eyeAlignmentBox
        if (showEyeAlignmentBox && boxToDraw != null) {
            // Apply scale and offset to the rect (boxToDraw is in image coordinates)
            val rect = RectF(
                boxToDraw.left * scaleFactor + offsetX,
                boxToDraw.top * scaleFactor + offsetY,
                boxToDraw.right * scaleFactor + offsetX,
                boxToDraw.bottom * scaleFactor + offsetY
            )
            canvas.drawRect(rect, boxPaint)
        }

        // Draw Focus Ring if active
        focusRing?.let { box ->
            val rect = RectF(
                box.left * scaleFactor + offsetX,
                box.top * scaleFactor + offsetY,
                box.right * scaleFactor + offsetX,
                box.bottom * scaleFactor + offsetY
            )
            canvas.drawRect(rect, focusPaint)
            
            // Draw crosshair
            val cx = rect.centerX()
            val cy = rect.centerY()
            val len = 20f
            canvas.drawLine(cx - len, cy, cx + len, cy, focusPaint)
            canvas.drawLine(cx, cy - len, cx, cy + len, focusPaint)
        }

        // Draw landmarks only if explicitly enabled
        if (!showLandmarks || results?.faceLandmarks().isNullOrEmpty()) {
            return
        }

        results?.let { faceLandmarkerResult ->
            faceLandmarkerResult.faceLandmarks().forEach { faceLandmarks ->
                drawFaceLandmarks(canvas, faceLandmarks, offsetX, offsetY)
                drawFaceConnectors(canvas, faceLandmarks, offsetX, offsetY)
            }
        }
    }

    private fun drawFaceLandmarks(
        canvas: Canvas,
        faceLandmarks: List<NormalizedLandmark>,
        offsetX: Float,
        offsetY: Float
    ) {
        faceLandmarks.forEach { landmark ->
            val x = landmark.x() * imageWidth * scaleFactor + offsetX
            val y = landmark.y() * imageHeight * scaleFactor + offsetY
            canvas.drawPoint(x, y, pointPaint)
        }
    }

    private fun drawFaceConnectors(
        canvas: Canvas,
        faceLandmarks: List<NormalizedLandmark>,
        offsetX: Float,
        offsetY: Float
    ) {
        FaceLandmarker.FACE_LANDMARKS_CONNECTORS.filterNotNull().forEach { connector ->
            val startLandmark = faceLandmarks.getOrNull(connector.start())
            val endLandmark = faceLandmarks.getOrNull(connector.end())

            if (startLandmark != null && endLandmark != null) {
                val startX = startLandmark.x() * imageWidth * scaleFactor + offsetX
                val startY = startLandmark.y() * imageHeight * scaleFactor + offsetY
                val endX = endLandmark.x() * imageWidth * scaleFactor + offsetX
                val endY = endLandmark.y() * imageHeight * scaleFactor + offsetY

                canvas.drawLine(startX, startY, endX, endY, linePaint)
            }
        }
    }

    fun setResults(
        faceLandmarkerResults: FaceLandmarkerResult,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: RunningMode = RunningMode.IMAGE
    ) {
        results = faceLandmarkerResults

        this.imageHeight = imageHeight
        this.imageWidth = imageWidth

        scaleFactor = when (runningMode) {
            RunningMode.IMAGE,
            RunningMode.VIDEO -> {
                min(width * 1f / imageWidth, height * 1f / imageHeight)
            }
            RunningMode.LIVE_STREAM -> {
                max(width * 1f / imageWidth, height * 1f / imageHeight)
            }
        }
        invalidate()
    }

    companion object {
        private const val LANDMARK_STROKE_WIDTH = 8F
        private const val TAG = "Face Landmarker Overlay"
    }
}
