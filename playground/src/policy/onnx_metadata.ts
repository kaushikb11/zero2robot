// Minimal reader for ONNX ModelProto.metadata_props.
//
// onnxruntime-web (1.27.0) does not expose custom model metadata through its
// public API, and the contract check must happen before we hand the model to
// the runtime anyway. metadata_props is field 14 of ModelProto, a repeated
// StringStringEntryProto { key = 1, value = 2 } — a ~50-line protobuf walk,
// which beats pulling in a protobuf dependency for two string fields.

const WIRE_VARINT = 0;
const WIRE_I64 = 1;
const WIRE_LEN = 2;
const WIRE_I32 = 5;

class Reader {
  pos = 0;
  constructor(private buf: Uint8Array) {}

  get done(): boolean {
    return this.pos >= this.buf.length;
  }

  varint(): number {
    let result = 0;
    let shift = 0;
    for (;;) {
      const b = this.buf[this.pos++];
      if (b === undefined) throw new Error('truncated protobuf');
      // String lengths fit well within 2^53 for any in-memory model.
      result += (b & 0x7f) * 2 ** shift;
      if ((b & 0x80) === 0) return result;
      shift += 7;
    }
  }

  /** Read a field tag varint, failing closed on overflow. Protobuf field tags
   *  are 32-bit (field_number << 3 | wire_type); a hostile file can encode a
   *  10-byte varint whose high bytes overflow. `tag >>> 3` (JS ToUint32) would
   *  silently truncate that to a plausible field/wire — e.g. fabricating
   *  field 14 / wire 2 and inventing metadata_props. Reject any tag that does
   *  not fit in 32 bits BEFORE it is split into field/wire. Capping the shift
   *  also keeps accumulation exact (result < 2^35 << 2^53), so no high byte is
   *  ever silently dropped by float rounding. */
  tag(): number {
    let result = 0;
    let shift = 0;
    for (;;) {
      const b = this.buf[this.pos++];
      if (b === undefined) throw new Error('truncated protobuf');
      if (shift >= 32) throw new Error('invalid protobuf tag: exceeds 32 bits');
      result += (b & 0x7f) * 2 ** shift;
      if ((b & 0x80) === 0) break;
      shift += 7;
    }
    if (result > 0xffffffff) throw new Error('invalid protobuf tag: exceeds 32 bits');
    return result;
  }

  bytes(len: number): Uint8Array {
    const out = this.buf.subarray(this.pos, this.pos + len);
    if (out.length !== len) throw new Error('truncated protobuf');
    this.pos += len;
    return out;
  }

  skip(wireType: number): void {
    if (wireType === WIRE_VARINT) this.varint();
    else if (wireType === WIRE_I64) this.pos += 8;
    else if (wireType === WIRE_I32) this.pos += 4;
    else if (wireType === WIRE_LEN) {
      // Two statements on purpose: `this.pos += this.varint()` captures the
      // OLD this.pos before varint() advances it (JS evaluates the left
      // operand of += first), silently corrupting the read offset.
      const len = this.varint();
      this.pos += len;
    } else throw new Error(`unsupported protobuf wire type ${wireType}`);
    if (this.pos > this.buf.length) throw new Error('truncated protobuf');
  }
}

const utf8 = new TextDecoder();

function parseStringStringEntry(buf: Uint8Array): [string, string] {
  const r = new Reader(buf);
  let key = '';
  let value = '';
  while (!r.done) {
    const tag = r.tag();
    const field = tag >>> 3;
    const wire = tag & 7;
    if (field === 1 && wire === WIRE_LEN) key = utf8.decode(r.bytes(r.varint()));
    else if (field === 2 && wire === WIRE_LEN) value = utf8.decode(r.bytes(r.varint()));
    else r.skip(wire);
  }
  return [key, value];
}

/** Extract metadata_props from serialized ONNX ModelProto bytes. */
export function readOnnxMetadata(modelBytes: Uint8Array): Map<string, string> {
  const meta = new Map<string, string>();
  const r = new Reader(modelBytes);
  while (!r.done) {
    const tag = r.tag();
    const field = tag >>> 3;
    const wire = tag & 7;
    if (field === 14 && wire === WIRE_LEN) {
      const [k, v] = parseStringStringEntry(r.bytes(r.varint()));
      meta.set(k, v);
    } else {
      r.skip(wire);
    }
  }
  return meta;
}
