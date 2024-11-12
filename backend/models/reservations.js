const mongoose = require('mongoose');
const { Schema } = mongoose;
const ReservationSchema = new Schema({
    userId: {
        type: Schema.Types.ObjectId, 
        ref: 'User',
        required: true
    },
    parkingId: {
        type: Schema.Types.ObjectId, 
        ref: 'Parking', 
        required: true
    },
    startTime: {
        type: Date,
        required: true
    },
    endTime: {
        type: Date // `endTime` no tiene `required: true` por lo que será opcional
    },
    status: {
        type: String,
        enum: ['active', 'completed', 'cancelled'], 
        required: true
    },
    totalCost: {
        type: Number,
        required: true
    }
}, {
    timestamps: true 
});


const Reservation = mongoose.model('Reservation', ReservationSchema);

module.exports = Reservation;
